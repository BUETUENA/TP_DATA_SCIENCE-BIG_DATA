#Algorithme implémenté par BUETUENA MALOZI JORDI, étudiant en Master 1 LMD, Data Science & IA; à l'Université de Kinshasa
#coding:utf-8
import math
import random

#Fonctions de distance 
def distance_euclidienne(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def distance_manhattan(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

#Prototypes 
def moyenne(points):
    n = len(points)
    dim = len(points[0])
    return [sum(p[d] for p in points) / n for d in range(dim)]

def mediane(points):
    dim = len(points[0])
    return [
        sorted(p[d] for p in points)[len(points) // 2]
        for d in range(dim)
    ]

#Algorithme des nuées dynamiques
class NueesDynamiques:
    def __init__(
        self,
        k=3,
        max_iter=300, #Nombre d'itérations maximales
        tol=1e-6,
        random_state=None,
        distance=distance_euclidienne,
        prototype_update=moyenne,
    ):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance = distance
        self.prototype_update = prototype_update

        self.prototypes_ = None
        self.labels_ = None
        self.critere_ = None

    def _initialiser(self, X):
        rng = random.Random(self.random_state)
        indices = rng.sample(range(len(X)), self.k)
        return [list(X[i]) for i in indices]

    def fit(self, X):
        X = [list(x) for x in X]
        n = len(X)

        if n < self.k:
            raise ValueError("k doit être <= au nombre de points.")

        self.prototypes_ = self._initialiser(X)
        labels = [-1] * n

        for _ in range(self.max_iter):
            nouveaux_labels = []
            for x in X:
                distances = [self.distance(x, p) for p in self.prototypes_]
                j = min(range(self.k), key=distances.__getitem__)
                nouveaux_labels.append(j)

            nouveaux_prototypes = []
            for j in range(self.k):
                cluster = [X[i] for i in range(n) if nouveaux_labels[i] == j]
                if cluster:
                    nouveaux_prototypes.append(self.prototype_update(cluster))
                else:
                    i_far = max(
                        range(n),
                        key=lambda i: self.distance(
                            X[i], self.prototypes_[nouveaux_labels[i]]
                        ),
                    )
                    nouveaux_prototypes.append(list(X[i_far]))

            deplacement = sum(
                self.distance(self.prototypes_[j], nouveaux_prototypes[j])
                for j in range(self.k)
            )

            self.prototypes_ = nouveaux_prototypes

            if nouveaux_labels == labels and deplacement <= self.tol:
                labels = nouveaux_labels
                break

            labels = nouveaux_labels

        self.labels_ = labels
        self.critere_ = sum(
            self.distance(X[i], self.prototypes_[labels[i]]) for i in range(n)
        )
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

#Saisie des valeurs par l'utilisateur

def demander_entier(message, minimum=1):
    while True:
        try:
            v = int(input(message))
            if v < minimum:
                print(f"Valeur >= {minimum} attendue.")
                continue
            return v
        except ValueError:
            print("Entier invalide, réessayez.")

def demander_flottant(message):
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Nombre invalide, réessayez.")

def saisir_points():
    n = demander_entier("Nombre de points à saisir : ", minimum=1)
    dim = demander_entier("Nombre de dimensions (ex: 2 pour (x,y)) : ", minimum=1)

    X = []
    print(f"\nSaisie des {n} points ({dim} coordonnées chacun) :")
    for i in range(n):
        print(f"\nPoint {i + 1}")
        point = [demander_flottant(f"  Coordonnée {d + 1} : ") for d in range(dim)]
        X.append(point)
    return X

def choisir_distance():
    print("\nChoix de la distance :")
    print("  1. Euclidienne  (prototype = moyenne)   -> k-means classique")
    print("  2. Manhattan    (prototype = médiane)   -> k-medians")
    while True:
        c = input("Votre choix (1/2) [1] : ").strip() or "1"
        if c == "1":
            return distance_euclidienne, moyenne, "Euclidienne / moyenne"
        if c == "2":
            return distance_manhattan, mediane, "Manhattan / médiane"
        print("Choix invalide.")

#Programme principal
def main():

    print("   ALGORITHME DES NUÉES DYNAMIQUES ")

    X = saisir_points()
    k = demander_entier(
        f"\nNombre de clusters k (2 <= k <= {len(X)}) : ", minimum=2
    )
    while k > len(X):
        print(f"k doit être <= {len(X)}.")
        k = demander_entier(f"Nombre de clusters k : ", minimum=2)

    dist_fn, proto_fn, nom_methode = choisir_distance()

    max_iter = demander_entier(
        "\nNombre max d'itérations [par défaut 300] : ", minimum=1
    )

    graine = input("Graine aléatoire (vide = aléatoire) : ").strip()
    random_state = int(graine) if graine else None

    print("\n\n")
    modele = NueesDynamiques(
        k=k,
        max_iter=max_iter,
        random_state=random_state,
        distance=dist_fn,
        prototype_update=proto_fn,
    )
    modele.fit(X)
   
    print(f"Méthode : {nom_methode}")

    for j in range(k):
        membres = [i for i, lab in enumerate(modele.labels_) if lab == j]
        print(f"\nCluster {j + 1}  (prototype = {modele.prototypes_[j]})")
        for i in membres:
            print(f"   • Point {i + 1} : {X[i]}")

    print("\nRécapitulatif")
    print("Labels       :", modele.labels_)
    print("Prototypes   :", modele.prototypes_)
    print(f"Critère total: {modele.critere_:.6f}")

if __name__ == "__main__":
    main()
