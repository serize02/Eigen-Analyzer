# Power Method: Numerical Analysis

The power method is an iterative technique used in numerical analysis to find the dominant eigenvalue and corresponding eigenvector of a matrix. It is particularly useful for large sparse matrices where other methods may be computationally expensive.  Its extension to the inverse power method is practical for finding any eigenvalue provided that a
good initial approximation is known.

## Computing the Dominant Eigenvalues

Let $A$ be an $n \times n$, non-singular real valued matrix with a basis of eigenvectors. Denote the eigenvalues by $\lambda_j$ and eigenvectors by $v_j$. 

We assume here there is a single eigenvalue of largest magnitude (the ‘dominant’ eigenvalue). Label them as follows:

$ |\lambda_1| > |\lambda_2| \geq |\lambda_3| \geq \dots \geq |\lambda_n| > 0$

Note that if $A$ has real-valued entries, it must be that $\lambda_1$ is real (why?).

The simplest approach to computing $\lambda_1$ and $v_1$ is the power method. The idea is as follows. Let $x$ be any vector. Then, since $\left\{v_j\right\}$ is a basis,

$x = c_1v_1+\dots+c_nv_n$

Now suppose that $c_1 \neq 0$. Then,

$Ax = c_1\lambda_1v_1 + \dots + c_n\lambda_nv_n$

and, applying $A$ again,

$A^kx = {\sum_{j=1}^n}c_j\lambda_j^kv_j$

Since the $\lambda_1^k$ term is largest in magnitude, in the sequence

$x, Ax, A^2x, A^3x\dots$

we expect the $\lambda_1^kv_1$  term will dominate so

$A^kx \approx c\lambda_1^kv_1 +$ smaller terms

Each iteration grows the largest term relative to the others, so after enough iterations only
the first term (what we want) will be left.

## Requirements

Ensure you have Python installed on your system. Then, install the necessary dependencies by running:

```shell
pip install -r requirements.txt


