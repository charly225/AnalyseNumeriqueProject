import numpy as np

def remontee_gauss(A, b):
    # Déterminons la taille n du systeme
    n = A.shape[0]

    # Initialisation du vecteur solution x
    x = np.zeros(n, dtype=float)

    if A[n-1, n-1] == 0:
        print("Erreur : L'élément diagonal A[n-1, n-1] est zéro, division impossible.")
        return None
    
    # Résolution du système par la méthode de remontée de Gauss
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        current_sum = 0.0
        for j in range(i+1, n):
            current_sum += A[i, j] * x[j]

        if A[i, i] == 0:
            print(f"Erreur : L'élément diagonal A[{i}, {i}] est zéro, division impossible.")
            return None    
        x[i] = (b[i] - current_sum) / A[i, i]

    return x

if __name__ == "__main__":
    # Exemple d'utilisation
    A = np.array([[2, -1, 0],
                  [0, 3, 1],
                  [0, 0, 4]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)

    solution = remontee_gauss(A, b)
    if solution is not None:
        print("La solution du système est :", solution)

    

