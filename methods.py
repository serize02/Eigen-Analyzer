import numpy as np
import pandas as pd

class Methods:
    @staticmethod
    def __augmented(coefficients, independents):
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

    @staticmethod
    def power_method(A, N, TOL):
        n = A.shape[0]
        x = np.ones(n)
        p = np.linalg.norm(x, ord=np.inf)
        x = x / p

        values = []

        for k in range(N):
            y = A.dot(x)
            u = p
            p = np.linalg.norm(y, ord=np.inf)
            err = np.linalg.norm(x - (y / p), ord=np.inf)
            x = y / p

            values.append(u)

            if err < TOL:
                return values

        raise ValueError("Failure after max number of iterations was reached")

    @staticmethod
    def inverse_power_method(A, N, TOL):
        n = A.shape[0]
        x = np.ones(n)
        q = x.T.dot(A).dot(x) / x.T.dot(x)  # initial approximation
        p = np.linalg.norm(x, ord=np.inf)
        x = np.ones(n) / p

        values = []

        for k in range(N):
            if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(Methods.__augmented(A, x)) and np.linalg.matrix_rank(
                    Methods.__augmented(A, x)) == A.shape[1]:
                return q, x

            y = np.linalg.solve(A - q * np.identity(n, float), x)
            u = p
            p = np.linalg.norm(y, ord=np.inf)
            err = np.linalg.norm(x - (y / p), ord=np.inf)
            x = y / p

            values.append(1 / u + q)

            if err < TOL:
                return values

        raise ValueError("Failure after max number of iterations was reached")

    @staticmethod
    def analyze_data(dataset, tolerance, iterations):
        """
        Loads the dataset from 'dataset.csv' and computes the eigenvalues using the inverse power method
        :return: None
        """
        df = dataset

        results = []
        failures = 0

        for index, row in df.iterrows():
            matrix = np.array(eval(row['matrix']))
            x = np.ones(matrix.shape[0])
            try:
                values_inverse = Methods.inverse_power_method(matrix, iterations, tolerance)
                values_power = Methods.power_method(matrix, iterations, tolerance)
                results.append({
                    'matrix': row['matrix'],
                    'power-method-values': values_power,
                    'inverse-power-method-values': values_inverse
                })
            except ValueError:
                failures += 1

        results_df = pd.DataFrame(results)

        return results_df