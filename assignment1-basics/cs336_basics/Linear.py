# write the Linear class for attnention mechanism
import numpy as np
class Linear:
    def __init__(self, input_dim, output_dim):
        # Initialize weights and bias
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        # Perform the linear transformation
        self.X = X  # Store input for backpropagation
        return np.dot(X, self.W) + self.b

    def backward(self, dY):
        # Compute gradients
        self.dW = np.dot(self.X.T, dY)
        self.db = np.sum(dY, axis=0, keepdims=True)
        dX = np.dot(dY, self.W.T)
        return dX

    def update(self, learning_rate):
        # Update weights and bias using gradients
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
        