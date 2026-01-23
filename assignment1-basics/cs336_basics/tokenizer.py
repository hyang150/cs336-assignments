"""
BPE (Byte Pair Encoding) Tokenizer Training

整体流程：
1. 并行预分词：把大文件切成小块，每个进程统计自己那块的 token 频率
2. 汇总结果：合并所有进程的统计结果
3. BPE 训练：循环找最频繁的字节对，合并，直到词表够大
"""

import multiprocessing
from collections import Counter, defaultdict
import regex as re


# GPT-2 的预分词正则表达式
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(file, num_chunks, separator):
    """
    找到文件的安全切分点，确保不会把 separator 切断

    Args:
        file: 已打开的二进制文件对象
        num_chunks: 想要切成几块
        separator: 分隔符（比如 b"<|endoftext|>"）

    Returns:
        boundaries: [0, pos1, pos2, ..., file_size]
    """
    file.seek(0, 2)  # 移到文件末尾
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // num_chunks
    boundaries = [0]

    for i in range(1, num_chunks):
        # 先跳到大概的位置
        target = i * chunk_size
        file.seek(target)

        # 往后找，直到遇到 separator 或文件末尾
        # 这样保证不会把一个单词切断
        buffer = file.read(len(separator) + 1000)  # 读一小段来找分隔符

        idx = buffer.find(separator)
        if idx != -1:
            # 找到了分隔符，边界设在分隔符之后
            boundaries.append(target + idx + len(separator))
        else:
            # 没找到，就用当前位置（可能不太理想，但能用）
            boundaries.append(min(target + len(buffer), file_size))

    boundaries.append(file_size)

    # 去重并排序
    boundaries = sorted(set(boundaries))
    return boundaries


def _worker_count_tokens(args):
    """
    工人进程的工作：
    1. 读取文件的指定区域
    2. 用正则切分成 tokens
    3. 统计每个 token 的出现次数

    Returns:
        Counter: {b'hello': 10, b'world': 5, ...}
                 注意：key 是 bytes 类型
    """
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # 解码成字符串
    text = chunk.decode("utf-8", errors="replace")

    # 用正则切分
    pieces = PAT.findall(text)

    # 转成 bytes 并统计（过滤掉 special tokens）
    special_set = set(special_tokens)
    token_counts = Counter()

    for piece in pieces:
        # 把每个 piece 转成 bytes
        piece_bytes = piece.encode("utf-8")
        if piece_bytes not in special_set:
            token_counts[piece_bytes] += 1

    return token_counts


def count_all_pairs(token_freqs):
    """
    统计所有 token 中相邻字节对的频率

    Args:
        token_freqs: {token_as_tuple: frequency, ...}
                     例如: {(104, 101, 108, 108, 111): 100}  # "hello" 出现 100 次

    Returns:
        Counter: {(byte1, byte2): count, ...}
    """
    pair_counts = Counter()

    for token_tuple, freq in token_freqs.items():
        # token_tuple 是一个字节元组，比如 (104, 101, 108, 108, 111)
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def merge_pair_in_tokens(token_freqs, pair_to_merge, new_token_id):
    """
    在所有 token 中把指定的 pair 合并成新的单个 token

    Args:
        token_freqs: {token_as_tuple: frequency}
        pair_to_merge: (byte1, byte2) 要合并的字节对
        new_token_id: 新 token 的 ID（整数）

    Returns:
        new_token_freqs: 更新后的 token 频率表
    """
    new_token_freqs = {}

    for token_tuple, freq in token_freqs.items():
        # 在这个 token 中找到所有的 pair_to_merge 并合并
        new_tuple = []
        i = 0
        while i < len(token_tuple):
            # 检查当前位置是否是要合并的 pair
            if (
                i < len(token_tuple) - 1
                and token_tuple[i] == pair_to_merge[0]
                and token_tuple[i + 1] == pair_to_merge[1]
            ):
                # 合并！用新的 token ID 替代这两个字节
                new_tuple.append(new_token_id)
                i += 2  # 跳过两个字节
            else:
                new_tuple.append(token_tuple[i])
                i += 1

        new_tuple = tuple(new_tuple)

        # 可能有多个不同的原始 token 合并后变成相同的
        if new_tuple in new_token_freqs:
            new_token_freqs[new_tuple] += freq
        else:
            new_token_freqs[new_tuple] = freq

    return new_token_freqs


def train_bpe(input_path, vocab_size, special_tokens):
    """
    主函数：训练 BPE tokenizer

    Args:
        input_path: 训练语料文件路径
        vocab_size: 目标词表大小
        special_tokens: 特殊 token 列表，如 ["<|endoftext|>"]

    Returns:
        vocab: {token_id: bytes} 词表
        merges: [(pair1, pair2), ...] 合并规则列表
    """
    print(f"开始训练 BPE，目标词表大小: {vocab_size}")

    # =====================================================
    # 第 1 步：并行预分词
    # =====================================================
    num_processes = multiprocessing.cpu_count()
    print(f"使用 {num_processes} 个进程并行预分词...")

    # 找到安全的切分边界
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    print(f"文件切分成 {len(boundaries) - 1} 块")

    # 准备每个工人的任务
    special_tokens_bytes = [
        s.encode("utf-8") if isinstance(s, str) else s for s in special_tokens
    ]

    worker_args = []
    for i in range(len(boundaries) - 1):
        worker_args.append(
            (input_path, boundaries[i], boundaries[i + 1], special_tokens_bytes)
        )

    # 并行执行
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(_worker_count_tokens, worker_args)

    # =====================================================
    # 第 2 步：汇总结果
    # =====================================================
    print("汇总各进程的统计结果...")

    total_counts_bytes = Counter()
    for res in results:
        total_counts_bytes.update(res)

    print(f"预分词完成，共 {len(total_counts_bytes)} 个不同的 token")

    # 把 bytes 转成 tuple 格式，方便后续合并操作
    # {b'hello': 100} -> {(104, 101, 108, 108, 111): 100}
    token_freqs = {}
    for token_bytes, freq in total_counts_bytes.items():
        token_tuple = tuple(token_bytes)  # bytes -> tuple of ints
        token_freqs[token_tuple] = freq

    # =====================================================
    # 第 3 步：初始化词表
    # =====================================================
    # 基础词表：256 个字节
    vocab = {i: bytes([i]) for i in range(256)}

    # 添加 special tokens
    next_id = 256
    for st in special_tokens:
        st_bytes = st.encode("utf-8") if isinstance(st, str) else st
        vocab[next_id] = st_bytes
        next_id += 1

    current_vocab_size = next_id
    merges = []

    print(f"初始词表大小: {current_vocab_size}")

    # =====================================================
    # 第 4 步：BPE 训练循环
    # =====================================================
    print("开始 BPE 合并循环...")

    iteration = 0
    while current_vocab_size < vocab_size:
        iteration += 1

        # A. 统计所有相邻字节对的频率
        pair_counts = count_all_pairs(token_freqs)

        if not pair_counts:
            print("没有更多可合并的字节对了")
            break

        # B. 找到频率最高的字节对
        best_pair, best_count = pair_counts.most_common(1)[0]

        # C. 创建新 token 并记录合并规则
        new_token_id = current_vocab_size

        # 新 token 的 bytes 表示
        # best_pair 可能包含之前合并产生的 token ID (>255)
        # 所以需要递归查找它们的 bytes 表示
        def get_bytes(token_id):
            if token_id < 256:
                return bytes([token_id])
            else:
                return vocab[token_id]

        new_token_bytes = get_bytes(best_pair[0]) + get_bytes(best_pair[1])
        vocab[new_token_id] = new_token_bytes
        merges.append(best_pair)

        # D. 更新 token_freqs：把所有的 best_pair 替换成 new_token_id
        token_freqs = merge_pair_in_tokens(token_freqs, best_pair, new_token_id)

        current_vocab_size += 1

        # 打印进度
        if iteration % 100 == 0:
            print(
                f"  迭代 {iteration}: 合并 {best_pair} -> {new_token_id}, "
                f"出现 {best_count} 次, 词表大小 {current_vocab_size}"
            )

    print(f"\n训练完成！")
    print(f"  最终词表大小: {len(vocab)}")
    print(f"  合并规则数量: {len(merges)}")

    return vocab, merges


# =====================================================
# 测试代码
# =====================================================
# if __name__ == "__main__":
#     pass
# 创建一个小的测试文件
# test_text = """Hello world! Hello everyone!
# The quick brown fox jumps over the lazy dog.
# Hello world! The world is beautiful.
# <|endoftext|>
# Another document here. Hello again!
# """ * 100  # 重复 100 次让它有点规模

# test_file = "/tmp/test_corpus.txt"
# with open(test_file, "w") as f:
#     f.write(test_text)

# # 训练 BPE
# vocab, merges = train_bpe(
#     input_path=test_file,
#     vocab_size=300,  # 256 + 1 special + 43 merges
#     special_tokens=["<|endoftext|>"]
# )

# # 打印前几个合并规则
# print("\n前 10 个合并规则:")
# for i, merge in enumerate(merges[:10]):
#     # 显示合并的是什么
#     def show(token_id):
#         if token_id < 256:
#             return repr(bytes([token_id]))
#         else:
#             return f"[{token_id}]={repr(vocab[token_id])}"

#     result_id = 256 + len(["<|endoftext|>"]) + i
#     print(f"  {i+1}. {show(merge[0])} + {show(merge[1])} -> {show(result_id)}")
