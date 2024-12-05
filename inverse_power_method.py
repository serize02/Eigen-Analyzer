import numpy as np
import pandas as pd

TOL = 1e-6
N = 20

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


def inverse_power_method (A, x):
    n = A.shape[0]
    q = x.T.dot(A).dot(x) / x.T.dot(x) # initial approximation
    p = np.linalg.norm(x, ord=np.inf) # smallest index of x with greater norm
    x = x / p

    values = []

    for k in range(N):
        if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(augmented(A, x)) and np.linalg.matrix_rank(augmented(A, x)) == A.shape[1] :
            return q, x

        y = np.linalg.solve(A - q * np.identity(n, float), x)
        u = p
        p = np.linalg.norm(y, ord=np.inf)
        err = np.linalg.norm(x - (y / p), ord=np.inf)
        x = y / p

        values.append(1/u+q)

        if err < TOL:
            return values, len(values)

    raise ValueError("Failure after max number of iterations was reached")

df = pd.read_csv('dataset.csv')

print("Dataset loaded")
print("size: ", df.size)

results = []

failures = 0

for index, row in df.iterrows():
    matrix = np.array( eval(row['matrix']))
    # x = np.random.rand(matrix.shape[0])
    x = np.ones(matrix.shape[0])
    try:
        values, iter = inverse_power_method(matrix, x)
        results.append({
            'dominant-eigenvalue': row['dominant-eigenvalue'],
            'computed-eigenvalues': values,
            'iterations': iter
        })
    except ValueError as e:
        failures+=1

results_df = pd.DataFrame(results)

results_df.to_csv('computed_results.csv', index=False)
print("Results saved to 'results.csv'")
print('Total failures: ', failures)