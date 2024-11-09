import numpy as np
import pandas as pd


def generate_symmetric_matrices_and_eigenvalues (num_matrices, low, high):
    data = []
    for size in range(2, 50):
        for _ in range(num_matrices):
            # Generate a random symmetric matrix
            A = np.random.randint(low, high, size = (size,size))
            matrix = (A + A.T) // 2 # Symmetric matrix
            eigenvalues, _ = np.linalg.eig(matrix)
            data.append({
                'Matrix': matrix.flatten(),
                'Eigenvalues': eigenvalues
            })
    return data

matrices_data = generate_symmetric_matrices_and_eigenvalues(100, 1, 100)

df = pd.DataFrame(matrices_data)

df['Matrix'] = df['Matrix'].apply(lambda x: np.array(x).reshape(int(np.sqrt(len(x))), int(np.sqrt(len(x)))).tolist())
df['Eigenvalues'] = df['Eigenvalues'].apply(lambda x: np.array(x).tolist())

df.to_csv('dataset.csv', index=False)
print("Dataset generated and saved to 'dataset.csv'")
