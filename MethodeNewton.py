import numpy as np
import matplotlib.pyplot as plt

def methode_newton(f, f_prime, x0, epsilon=1e-10, max_iter=100, afficher=True):
    """
    Méthode de Newton pour résoudre f(x) = 0
    
    Formule : x_{n+1} = x_n - f(x_n) / f'(x_n)
    
    Conditions d'utilisation :
    - f doit être C² (dérivable 2 fois avec dérivées continues)
    - f'(x_n) ≠ 0 pour tous les n
    - x0 doit être au voisinage de la solution
    
    Convergence : quadratique (doublement du nombre de décimales à chaque itération)
    """
    x = x0
    historique = [x0]
    erreurs = []
    
    for n in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        
        # Vérification : dérivée non nulle
        if abs(fpx) < 1e-15:
            print(f"⚠ Arrêt : dérivée nulle à l'itération {n}")
            return None, n, historique, erreurs
        
        # Formule de Newton
        x_nouveau = x - fx / fpx
        historique.append(x_nouveau)
        erreurs.append(abs(x_nouveau - x))
        
        if afficher and n < 10:
            print(f"Itération {n+1:2d} : x = {x_nouveau:.15f}  |Δx| = {erreurs[-1]:.2e}")
        
        # Critère d'arrêt
        if erreurs[-1] < epsilon:
            if afficher:
                print(f"\n✓ Convergence atteinte en {n+1} itérations")
            return x_nouveau, n+1, historique, erreurs
        
        x = x_nouveau
    
    print(f"⚠ Nombre maximum d'itérations atteint")
    return x, max_iter, historique, erreurs


def visualiser_convergence(historique, erreurs, racine_exacte=None, titre=""):
    """Visualise la convergence de la méthode de Newton"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = range(len(historique))
    
    # Graphique 1 : Évolution de x_n
    ax1.plot(iterations, historique, 'bo-', linewidth=2, markersize=6)
    if racine_exacte is not None:
        ax1.axhline(y=racine_exacte, color='r', linestyle='--', 
                    label=f'Racine exacte = {racine_exacte:.6f}')
        ax1.legend()
    ax1.set_xlabel('Itération n', fontsize=12)
    ax1.set_ylabel('$x_n$', fontsize=12)
    ax1.set_title('Convergence vers la racine', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2 : Erreur en échelle logarithmique
    if erreurs:
        ax2.semilogy(range(1, len(erreurs)+1), erreurs, 'ro-', 
                     linewidth=2, markersize=6)
        ax2.set_xlabel('Itération n', fontsize=12)
        ax2.set_ylabel('$|x_{n+1} - x_n|$ (échelle log)', fontsize=12)
        ax2.set_title('Vitesse de convergence (quadratique)', fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(titre, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==================== EXEMPLES D'UTILISATION ====================

print("=" * 70)
print(" " * 15 + "MÉTHODE DE NEWTON - EXEMPLES")
print("=" * 70)

# EXEMPLE 1 : Calcul de √2
print("\n📌 EXEMPLE 1 : Calcul de √2")
print("-" * 70)
print("Résolution de f(x) = x² - 2 = 0\n")

def f1(x):
    return x**2 - 2

def f1_prime(x):
    return 2*x

x0 = 1.5
racine, nb_iter, hist, err = methode_newton(f1, f1_prime, x0, epsilon=1e-12)

if racine is not None:
    print(f"\nRésultat    : {racine}")
    print(f"Valeur √2   : {np.sqrt(2)}")
    print(f"Erreur abs  : {abs(racine - np.sqrt(2)):.2e}")
    visualiser_convergence(hist, err, np.sqrt(2), "Exemple 1 : √2")

# EXEMPLE 2 : Résolution de cos(x) = x
print("\n\n📌 EXEMPLE 2 : Résolution de cos(x) = x")
print("-" * 70)
print("Résolution de f(x) = cos(x) - x = 0\n")

def f2(x):
    return np.cos(x) - x

def f2_prime(x):
    return -np.sin(x) - 1

x0 = 0.5
racine2, nb_iter2, hist2, err2 = methode_newton(f2, f2_prime, x0, epsilon=1e-12)

if racine2 is not None:
    print(f"\nRacine trouvée : {racine2}")
    print(f"Vérification : cos({racine2}) = {np.cos(racine2)}")
    visualiser_convergence(hist2, err2, racine2, "Exemple 2 : cos(x) = x")

print("\n" + "=" * 70)
print("✓ Implémentation terminée avec succès !")
print("=" * 70)