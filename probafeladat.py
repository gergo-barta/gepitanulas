"""
Feladatok

    Olvassa be a digits beépített adatállományt és írassa ki a legfontosabb jellemzőit (rekordok száma, attribútumok száma és osztályok száma). (3 pont)
    Készítsen többdimenziós vizualizációt a mátrix ábra segítségével (pairplot). (4 pont)
    Particionálja az adatállományt 80% tanító és 20% tesztállományra. Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (3 pont)
    Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: döntési fa (4 mélység, entrópia homogenitási kritérium), logisztikus regresszió (liblinear solverrel) és neurális háló (1 rejtett réteg 4 neuronnal, logisztikus aktivációs függvény). A teszt score alapján hasonlítsa össze az illesztett modelleket, melyeket nyomtasson ki. (10 pont)
    Számolja ki az 5. pont legjobb modelljére a teszt tévesztési mátrixot. (4 pont)
    Ábrázolja a tévesztési mátrixot. (3 pont)
    Végezzen nemfelügyelt tanítást a K-közép módszerrel az input attribútumokon. Határozza meg az optimális klaszterszámot 30-ig a DB indexszel. Az optimális klaszterszám mellett vizualizálja a klasztereket egy pontdiagrammon, ahol a két koordináta egy 2 dimenziós PCA eredménye. (13 pont)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Adatok beolvasása és jellemzők kiíratása (3 pont)
# ---------------------------------------------------------
digits = load_digits()
X = digits.data
y = digits.target

print("1. FELADAT: Adathalmaz jellemzői")
print(f"Rekordok száma: {X.shape[0]}")
print(f"Attribútumok száma: {X.shape[1]}") # 64 pixel (8x8 képek)
print(f"Osztályok száma: {len(np.unique(y))}") # 0-tól 9-ig
print("-" * 40)

# ---------------------------------------------------------
# 2. Többdimenziós vizualizáció - Mátrix ábra / Pairplot (4 pont)
# ---------------------------------------------------------
# FONTOS: A digits adathalmaznak 64 attribútuma van. 
# Egy 64x64-es pairplot renderelése órákig tartana és olvashatatlan lenne.
# Ezért a vizualizációhoz kiválasztjuk az első 5 érdemi oszlopot.

print("2. FELADAT: Mátrix ábra készítése (az első 5 feature alapján)...")

# DataFrame készítése a vizualizációhoz
df_viz = pd.DataFrame(X[:, :5], columns=[f'pixel_{i}' for i in range(5)])
df_viz['target'] = y

plt.figure(figsize=(10, 8))
# A seaborn pairplot felel meg a mátrix ábrának
sns.pairplot(df_viz, hue='target', palette='tab10')
plt.suptitle("Mátrix ábra (Pairplot) az első 5 pixelre", y=1.02)
plt.show()

# ---------------------------------------------------------
# 3. Particionálás (3 pont)
# ---------------------------------------------------------
# 80% tanító, 20% teszt, random_state=2025
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=2025
)

print(f"3. FELADAT: Adatok szétválasztva (Train: {len(X_train)}, Test: {len(X_test)})")
print("-" * 40)

# ---------------------------------------------------------
# 4. Felügyelt tanítás és összehasonlítás (10 pont)
# ---------------------------------------------------------
results = {}
models = {}

# A) Döntési fa (4 mélység, entrópia)
dt = DecisionTreeClassifier(
    max_depth=4, 
    criterion='entropy', 
    random_state=2025
)
dt.fit(X_train, y_train)
models['Decision Tree'] = dt
results['Decision Tree'] = dt.score(X_test, y_test)

# B) Logisztikus regresszió (liblinear solver)
lr = LogisticRegression(
    solver='liblinear', 
    random_state=2025,
    max_iter=1000 # Növeljük, hogy biztosan konvergáljon
)
lr.fit(X_train, y_train)
models['Logistic Regression'] = lr
results['Logistic Regression'] = lr.score(X_test, y_test)

# C) Neurális háló (1 rejtett réteg 4 neuronnal, logisztikus aktiváció)
mlp = MLPClassifier(
    hidden_layer_sizes=(4,),  # Egy tuple: (4,) jelent egy réteget 4 neuronnal
    activation='logistic', 
    max_iter=2000,            # Neurális hálónak kell idő a tanuláshoz
    random_state=2025
)
mlp.fit(X_train, y_train)
models['Neural Network'] = mlp
results['Neural Network'] = mlp.score(X_test, y_test)

print("4. FELADAT: Modellek pontossága (Accuracy):")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Legjobb modell kiválasztása
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n=> A legjobb modell: {best_model_name}")
print("-" * 40)

# ---------------------------------------------------------
# 5. és 6. Tévesztési mátrix számítása és ábrázolása (4 + 3 pont)
# ---------------------------------------------------------
print(f"5-6. FELADAT: Tévesztési mátrix a(z) {best_model_name} modellhez")

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Ábrázolás
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Tévesztési Mátrix: {best_model_name}')
plt.show()

# ---------------------------------------------------------
# 7. Nemfelügyelt tanítás (K-Means + DB Index + PCA) (13 pont)
# ---------------------------------------------------------
print("7. FELADAT: K-Means klaszterezés optimalizálása...")

# Érdemes skálázni az adatokat klaszterezés előtt
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db_scores = []
K_range = range(2, 31) # 2-től 30-ig

for k in K_range:
    # random_state rögzítése a reprodukálhatóságért
    kmeans = KMeans(n_clusters=k, random_state=2025, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # DB index számítása
    score = davies_bouldin_score(X_scaled, labels)
    db_scores.append(score)

# Optimális K keresése (Minimum DB index)
optimal_k_idx = np.argmin(db_scores)
optimal_k = K_range[optimal_k_idx]
min_db_score = db_scores[optimal_k_idx]

print(f"Vizsgált tartomány: 2-30. Optimális klaszterszám: {optimal_k}")
print(f"Legjobb DB index érték: {min_db_score:.4f}")

# Végső modell az optimális K-val
final_kmeans = KMeans(n_clusters=optimal_k, random_state=2025, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# PCA vizualizáció (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab20', s=20)
plt.title(f"K-Means klaszterek (k={optimal_k}) PCA vizualizációval")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label='Cluster ID')
plt.show()

 
