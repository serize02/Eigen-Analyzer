import numpy as np

def read_matrix(matrix_dim):
    """
    Reads a nxn matrix
    :param matrix_dim: Matrix dimension
    :return: Matrix of dimension matrix_dim x matrix_dim
    """
    temp = np.zeros((matrix_dim, matrix_dim), float)

    for row in range(matrix_dim):
        for column in range(matrix_dim):
            temp[row][column] = float(input())

    return temp


def read_vector(vector_dim):
    """
    Reads a n-dimension vector
    :param vector_dim: Vector dimension
    :return: Vector of dimension vector_dim
    """
    temp = np.zeros(vector_dim, float)

    for index in range(vector_dim):
        temp[index] = float(input())

    return temp

def infinite_norm(vector):
    """
    Returns the infinite norm of vector and its smallest index
    :param vector: To calculate infinite norm of
    :return: Norm of vector and its smallest index
    """
    maximum, smallest_i = abs(vector[0]), 0

    for index in range(vector.shape[0]):
        if abs(vector[index]) > maximum:
            maximum, smallest_i = abs(vector[index]), index

    return maximum, smallest_i


def augmented(coefficients, independents):
    """
    Returns augmented matrix
    :param coefficients: Coefficients of the equation
    :param independents: Independent values of the equation
    :return: The augmented matrix formed by coefficients and independent values
    """
    temp = np.zeros((coefficients.shape[0], coefficients.shape[0] + 1), float)

    for row in range(coefficients.shape[0]):
        temp[row][coefficients.shape[0]] = independents[row]

    return temp


# Get matrix dimensions
n = int(input())
# Read nxn matrix
A = read_matrix(n)
# Read the initial vector
x = read_vector(n)
# Get maximum number of iterations
N = int(input())
# Get error tolerance
tol = float(input())

# Set initial approximation 'q'
q = x.T.dot(A).dot(x) / x.T.dot(x)
# Smallest index of x with greater norm
p = infinite_norm(x)[1]
x = x / x[p]
k = 1

while k <= N:
    if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(augmented(A, x)) and np.linalg.matrix_rank(augmented(A, x)) == A.shape[1] :
        # The linear system has infinite solutions
        print(q, " is an eigenvalue")
        break
    y = np.linalg.solve(A - q * np.identity(n, float), x)
    u = y[p]
    # Smallest index of y with greater norm
    p = infinite_norm(y)[1]
    err = infinite_norm(x - (y / y[p]))[0]
    x = y / y[p]

    if err < tol:
        print(1/u + q, " is an eigenvalue\n")
        print(x, " is its associated eigenvector")
        break

    k += 1

if k > N:
    print("Failure after ", N, " iterations")
