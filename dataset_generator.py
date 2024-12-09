import numpy as np
import pandas as pd

def generate_symmetric_matrices_and_eigenvalues (num_matrices, low, high):
    data = []
    for size in range(20,30):
        for _ in range(num_matrices):
            # Generate a random symmetric matrix
            A = np.random.randint(low, high, size = (size,size))
            matrix = (A + A.T) // 2 # Symmetric matrix
            eigenvalues, _ = np.linalg.eig(matrix)
            dominant = np.max(np.abs(eigenvalues))
            data.append({
                'matrix': matrix.flatten(),
                'dominant-eigenvalue': dominant
            })
    return data

matrices_data = generate_symmetric_matrices_and_eigenvalues(10, 1, 100)

df = pd.DataFrame(matrices_data)

df['matrix'] = df['matrix'].apply(lambda x: np.array(x).reshape(int(np.sqrt(len(x))), int(np.sqrt(len(x)))).tolist())
df['dominant-eigenvalue'] = df['dominant-eigenvalue'].apply(lambda x: round(x, 7))

df.to_csv('large_dataset.csv', index=False)
print("Dataset generated and saved to 'dataset.csv'")