import numpy as np

def sigmoid(z):
    """
    Function to calculate sigmoid
    """
    return 1/(1 + np.exp(-z))

def gradient_descent(X, y, weights, learning_rate, iterations, e):
    """
    Function to perform gradient descent and update the weights accordingly
    """
    m = len(y)
    print(f"X is {X}")
    print(f"y is {y}")

    for _ in range(iterations):

        # Computes the linear combination of input features (𝑋) and weights (𝜃), resulting in a vector 𝑧.
        # The weights vector/matrix assigns importance to each feature in 𝑋.
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        print(f"predictions {_} are {predictions}")

        error = predictions - y
        print(f"error {_} is {error}")

        gradient = np.dot(X.T, error)/m
        print(f"gradient {_} is {gradient}")

        gradient_magnitude = np.linalg.norm(gradient)
        print(f"gradient_magnitude {_} is {gradient_magnitude}")

        if gradient_magnitude < e:
            print("the gradient vector's magnitude is equal to 0.01 and has converged")
            break
        
        # This updates the weights by moving them slightly in the direction opposite to the gradient. 
        # This is because we want to reduce the loss, not increase it.
        weights -= learning_rate * gradient
        print(f"weights {_} is {weights}")
        print("==========================")

        
        
    return weights
m = 5
n = 3

X_var = np.random.randn(m,n) # Random distribution 0 or 1 for a m x n input matrix
weight_var = np.random.randn(n, 1) # Random distribution 0 or 1 for a n x 1 weight matrix

# Generate a target vector (y) with m examples (binary classification: 0 or 1)
y_var = np.random.randint(0, 2, size=(m, 1)) # Random integers 0 or 1 column vector 


gradient_descent(X=X_var,
                 weights=weight_var,
                 y=y_var,
                 learning_rate=0.01,
                 iterations= 5,
                 e=0.001)
