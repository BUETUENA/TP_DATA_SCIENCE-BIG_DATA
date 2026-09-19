# -*- coding: utf-8 -*-
import math
import random

# 1. Opérations vectorielles / matricielles de base
def v_add(u, v):  return [a + b for a, b in zip(u, v)]
def v_sub(u, v):  return [a - b for a, b in zip(u, v)]
def v_scale(c, v): return [c * x for x in v]
def v_dot(u, v):  return sum(a * b for a, b in zip(u, v))
def v_norm(v):    return math.sqrt(sum(x * x for x in v))
def v_dist(a, b): return v_norm(v_sub(a, b))


def v_mean(vectors):
    n, d = len(vectors), len(vectors[0])
    return [sum(v[j] for v in vectors) / n for j in range(d)]


def v_cov(vectors, mu=None):
    """Matrice de covariance (estimateur 1/(n-1))."""
    n, d = len(vectors), len(vectors[0])
    if mu is None:
        mu = v_mean(vectors)
    C = [[0.0] * d for _ in range(d)]
    for v in vectors:
        diff = v_sub(v, mu)
        for i in range(d):
            for j in range(d):
                C[i][j] += diff[i] * diff[j]
    denom = max(1, n - 1)
    for i in range(d):
        for j in range(d):
            C[i][j] /= denom
    return C


def mat_inv(A):
    """Inverse par Gauss-Jordan (avec régularisation si singulier)."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda k: abs(M[k][i]))
        if abs(M[piv][i]) < 1e-12:
            M[i][i] += 1e-6
            continue
        M[i], M[piv] = M[piv], M[i]
        p = M[i][i]
        for j in range(2 * n):
            M[i][j] /= p
        for k in range(n):
            if k != i:
                f = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= f * M[i][j]
    return [row[n:] for row in M]


def jacobi_eig(A, max_iter=200, tol=1e-12):
    """Diagonalisation de Jacobi d'une matrice symétrique."""
    n = len(A)
    A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_iter):
        off, p, q = 0.0, 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > off:
                    off = abs(A[i][j]); p, q = i, j
        if off < tol:
            break
        theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
        t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
        c = 1 / math.sqrt(t * t + 1)
        s = t * c
        for i in range(n):
            if i != p and i != q:
                aip, aiq = A[i][p], A[i][q]
                A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p]
                A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q]
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        A[p][q] = A[q][p] = 0.0
        for i in range(n):
            vip, viq = V[i][p], V[i][q]
            V[i][p] = c * vip - s * viq
            V[i][q] = s * vip + c * viq
    vals = [A[i][i] for i in range(n)]
    vecs = [[V[i][j] for i in range(n)] for j in range(n)]  # vecs[j] = j-ème colonne
    return vals, vecs



# 2. Classe principale : Nuées Dynamiques
class NueeDynamique:
    """
    data        : liste de vecteurs (individus)
    K           : nombre de classes
    nuee_type   : 'point' | 'set' | 'axes' | 'distribution' | 'structure'
    n1          : nombre d'étalons (uniquement pour 'set')
    p           : dimension du sous-espace (pour 'axes'/'structure')
    """

    def __init__(self, data, K, nuee_type='point', n1=1, p=1):
        self.data = data
        self.N = len(data)
        self.M = len(data[0])
        self.K = K
        self.nuee_type = nuee_type
        self.n1 = min(n1, self.N)
        self.p = min(p, self.M)

        self.reps = [None] * K       # K nuées (représentations)
        self.classes = [[] for _ in range(K)]

    # ---------- D(x, E) : distance d'un point à une nuée ----------
    def dist_to_rep(self, x, rep):
        if rep is None:
            return float('inf')
        t = self.nuee_type

        if t == 'point':
            return v_dist(x, rep)

        if t == 'set':
            if not rep:
                return float('inf')
            # distance moyenne aux étalons (robuste)
            return sum(v_dist(x, y) for y in rep) / len(rep)

        if t in ('axes', 'structure'):
            mu, basis = rep
            diff = v_sub(x, mu)
            for v in basis:
                diff = v_sub(diff, v_scale(v_dot(diff, v), v))
            return v_norm(diff)

        if t == 'distribution':
            mu, inv = rep
            diff = v_sub(x, mu)
            tmp = [sum(inv[i][j] * diff[j] for j in range(self.M))
                   for i in range(self.M)]
            return math.sqrt(max(0.0, v_dot(diff, tmp)))

        return float('inf')

    # ---------- Mise à jour d'une nuée à partir de ses membres ----------
    def compute_rep(self, class_idx):
        if not class_idx:
            return None
        pts = [self.data[i] for i in class_idx]
        t = self.nuee_type

        # --- (1) Point unique : centroïde ---
        if t == 'point':
            return v_mean(pts)

        # --- (2) Ensemble d'étalons : n1 points de E minimisant R(x,i,L) ---
        if t == 'set':
            scores = []
            for idx in range(self.N):
                x = self.data[idx]
                d = sum(v_dist(x, self.data[j]) for j in class_idx) / len(class_idx)
                scores.append((d, idx))
            scores.sort(key=lambda z: z[0])
            return [self.data[idx] for _, idx in scores[:self.n1]]

        # --- (3) / (5) Axes factoriels / sous-espace affine ---
        if t in ('axes', 'structure'):
            mu = v_mean(pts)
            if len(pts) < 2:
                return (mu, [])
            cov = v_cov(pts, mu)
            vals, vecs = jacobi_eig(cov)
            order = sorted(range(len(vals)), key=lambda i: -vals[i])
            p = min(self.p, self.M, len(order))
            basis = [vecs[i] for i in order[:p]]
            return (mu, basis)

        # --- (4) Distribution de probabilité (gaussienne) ---
        if t == 'distribution':
            mu = v_mean(pts)
            if len(pts) < 2:
                cov = [[1.0 if i == j else 0.0 for j in range(self.M)]
                       for i in range(self.M)]
            else:
                cov = v_cov(pts, mu)
            for i in range(self.M):
                cov[i][i] += 1e-6
            return (mu, mat_inv(cov))

        return None

    # ---------- Étape 2 de l'algorithme : partition ----------
    def assign(self):
        cls = [[] for _ in range(self.K)]
        for idx, x in enumerate(self.data):
            best_k, best_d = 0, float('inf')
            for k in range(self.K):
                d = self.dist_to_rep(x, self.reps[k])
                if d < best_d:
                    best_d, best_k = d, k
            cls[best_k].append(idx)
        return cls

    # ---------- Critère S(L, L) ----------
    def compute_S(self):
        total = 0.0
        for k in range(self.K):
            c = self.classes[k]
            if not c:
                continue
            for idx in c:
                x = self.data[idx]
                d = sum(v_dist(x, self.data[j]) for j in c) / len(c)
                total += d
        return total

    # ---------- Initialisation ----------
    def initialize_random(self):
        idx = list(range(self.N))
        random.shuffle(idx)
        self.classes = [[] for _ in range(self.K)]
        for i, j in enumerate(idx):
            self.classes[i % self.K].append(j)
        self.reps = [self.compute_rep(c) for c in self.classes]

    def initialize_manual(self, initial_reps):
        self.reps = initial_reps
        self.classes = self.assign()

    # ---------- Boucle principale ----------
    def run(self, max_iter=100, verbose=True):
        history = []
        for it in range(max_iter):
            old_classes = [c[:] for c in self.classes]
            # Étape 1 : affectation
            self.classes = self.assign()
            # Étape 2 : mise à jour des nuées
            for k in range(self.K):
                if self.classes[k]:
                    self.reps[k] = self.compute_rep(self.classes[k])
            # Critère
            S = self.compute_S()
            history.append(S)
            if verbose:
                sizes = [len(c) for c in self.classes]
                print(f"  Itération {it + 1:3d} : S = {S:.6f}   tailles = {sizes}")
            # Test de convergence
            if old_classes == self.classes:
                if verbose:
                    print(f"  >> Convergence atteinte à l'itération {it + 1}")
                break
        return history


# 3. Interface clavier
def input_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            v = int(input(prompt))
            if min_val is not None and v < min_val:
                print(f"  Valeur >= {min_val} attendue.")
                continue
            if max_val is not None and v > max_val:
                print(f"  Valeur <= {max_val} attendue.")
                continue
            return v
        except ValueError:
            print("  Entrée invalide. Entrez un entier.")


def input_vector(prompt, d):
    while True:
        s = input(prompt)
        try:
            vals = [float(x) for x in s.replace(',', ' ').split()]
            if len(vals) != d:
                print(f"  Il faut exactement {d} valeurs.")
                continue
            return vals
        except ValueError:
            print("  Entrée invalide. Entrez des nombres séparés par des espaces.")

# 4. Programme principal


def main():
    print()
    print("" )
    print("  MÉTHODE DES NUÉES DYNAMIQUES  (E. Diday, 1971)")
    print("" )

    # ---------- Données ----------
    print("\n--- Saisie des données ---")
    N = input_int("Nombre d'individus N : ", 2)
    M = input_int("Nombre de paramètres (dimension) M : ", 1)
    print(f"Entrez les {N} individus ({M} valeur(s) par individu) :")
    data = [input_vector(f"  x{i + 1} : ", M) for i in range(N)]

    # ---------- Paramètres ----------
    print("\n--- Paramètres de l'algorithme ---")
    K = input_int(f"Nombre de classes K (2 <= K <= {N}) : ", 2, N)

    print("\nType de nuée (représentation des classes) :")
    print("  1. Point unique                (équivalent k-means)")
    print("  2. Ensemble de points          (nuée d'étalons)")
    print("  3. Axes factoriels             (1 axe principal)")
    print("  4. Distribution de probabilité (gaussienne)")
    print("  5. Structure représentative    (sous-espace affine)")
    t = input_int("Votre choix : ", 1, 5)

    if t == 1:
        nuee_type, n1, p = 'point', 1, 1
    elif t == 2:
        nuee_type = 'set'
        n1 = input_int(f"Nombre d'étalons n1 par classe (1 <= n1 <= {N}) : ", 1, N)
        p = 1
    elif t == 3:
        nuee_type, n1, p = 'axes', 1, 1
    elif t == 4:
        nuee_type, n1, p = 'distribution', 1, 1
    else:
        nuee_type = 'structure'
        p = input_int(f"Dimension du sous-espace (1 <= p <= {M}) : ", 1, M)
        n1 = 1

    print("\nMode d'initialisation :")
    print("  1. Aléatoire (recommandé)")
    print("  2. Avec graine fixe (reproductible)")
    seed = input_int("Votre choix : ", 1, 2)
    if seed == 2:
        random.seed(12345)

    # ---------- Exécution ----------
    print("\n--- Exécution de l'algorithme ---")
    algo = NueeDynamique(data, K, nuee_type, n1=n1, p=p)
    algo.initialize_random()
    history = algo.run(max_iter=100, verbose=True)

    # ---------- Résultats ----------
    print("\n" + "" )
    print("  RÉSULTATS")
    print("")
    for k in range(K):
        print(f"\n--- Classe {k + 1} : {len(algo.classes[k])} individu(s) ---")
        print(f"  Indices : {[i + 1 for i in algo.classes[k]]}")
        rep = algo.reps[k]
        if rep is None:
            print("  (classe vide)")
            continue
        if nuee_type == 'point':
            print(f"  Centre : {[round(v, 4) for v in rep]}")
        elif nuee_type == 'set':
            print(f"  {len(rep)} étalon(s) :")
            for e in rep:
                print(f"    {[round(v, 4) for v in e]}")
        elif nuee_type in ('axes', 'structure'):
            mu, basis = rep
            print(f"  Centre : {[round(v, 4) for v in mu]}")
            print(f"  Base du sous-espace ({len(basis)} vecteur(s)) :")
            for v in basis:
                print(f"    {[round(x, 4) for x in v]}")
        elif nuee_type == 'distribution':
            mu, inv = rep
            print(f"  Moyenne : {[round(v, 4) for v in mu]}")
            print(f"  Matrice de covariance inverse :")
            for row in inv:
                print(f"    {[round(x, 4) for x in row]}")

    print(f"\nValeur finale S(L,L) = {history[-1]:.6f}")
    print(f"Nombre d'itérations : {len(history)}")
 

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nInterrompu par l'utilisateur.")