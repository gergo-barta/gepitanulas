"""
A labor teljesítése egy dolgozat sikeres megírásával történik az utolsó laboron.  A dolgozat szerkezete az alábbi:

    Olvassa be az adatállományt egy URL címről, ahol az első sor tartalmazza az attribútum neveket. (3 pont)
    Készítsen dataframe-t majd nyomtassa ki az alábbi leíró statisztikákat (pl. átlag, szórás a célváltozó szerint csoportosítva). (4 pont)
    Készítsen többdimenziós vizualizációt: matrix-plot, Andrews ábra, párhuzamos tengelyek közül valamelyik. (4 pont)
    Particionálja az adatállományt 80% tanító és 20% tesztállományra. Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (2 pont)
    Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: döntési fa (mélység, levelek vagy vágások mérete, homogenitási kritérium beállítása), logisztikus regresszió (solver és iteráció beállítása), neurális háló (a háló mérete, az aktivációs függvény megválasztása). A score alapján hasonlítsa össze az illesztett modelleket. (10 pont)
    Számolja ki az 5. pont legjobb modelljére a tévesztési mátrixot és ábrázolja a ROC-görbét az AUC értékkel. (4 pont)
    Végezzen nemfelügyelt tanítást a K-közép módszerrel az input attribútumokon. Határozza meg az optimális klaszterszámot 50-ig a DB indexszel. Az optimális klaszterszám mellett vizualizálja a klasztereket egy pontdiagrammon, ahol a két koordináta egy 2 dimenziós PCA eredménye. (13 pont)
    added blabla
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, andrews_curves

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------
# 1. Adatbeolvasás URL-ről (3 pont)
# ---------------------------------------------------------
# Példa URL (Iris adathalmaz)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Célváltozó és attribútumok szétválasztása (feltételezzük, hogy az utolsó oszlop a cél)
target_col = df.columns[-1]  # 'species'
feature_cols = df.columns[:-1]

# !!! A 6. feladat (ROC) miatt most egyszerűsítjük binárisra az adatot a példa kedvéért !!!
# A vizsgán ezt csak akkor tedd meg, ha kérik, vagy ha az adathalmaz alapból bináris.
# Itt most eldobjuk a 'setosa' osztályt, hogy csak 2 maradjon (versicolor vs virginica).
df = df[df[target_col] != 'setosa']

print("1. Adatok beolvasva. Első 5 sor:")
print(df.head())
print("-" * 30)

# ---------------------------------------------------------
# 2. Leíró statisztikák (4 pont)
# ---------------------------------------------------------
print("2. Leíró statisztikák a célváltozó szerint csoportosítva:")
# Átlag és szórás kiszámítása
stats = df.groupby(target_col)[feature_cols].agg(['mean', 'std'])
print(stats)
print("-" * 30)

# ---------------------------------------------------------
# 3. Többdimenziós vizualizáció (4 pont)
# ---------------------------------------------------------
# Párhuzamos tengelyek (Parallel Coordinates) vagy Andrews ábra
plt.figure(figsize=(10, 6))
# A normalizálás segít a szebb ábrázolásban, de a beépített függvények kezelik az alapokat
parallel_coordinates(df, target_col, colormap='viridis')
plt.title('3. Feladat: Párhuzamos tengelyek (Parallel Coordinates)')
plt.show()

# ---------------------------------------------------------
# 4. Particionálás (Train/Test Split) (2 pont)
# ---------------------------------------------------------
X = df[feature_cols]
y = df[target_col]

# Célváltozó kódolása számokra (szükséges a modellekhez)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Random state = idei év (2025)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, shuffle=True, random_state=2025
)

print(f"4. Adatok szétválasztva. Train méret: {X_train.shape}, Test méret: {X_test.shape}")
print("-" * 30)

# ---------------------------------------------------------
# 5. Felügyelt tanítás (10 pont)
# ---------------------------------------------------------
results = {}

# A) Döntési fa
dt = DecisionTreeClassifier(
    max_depth=5,           # Mélység
    min_samples_leaf=5,    # Levelek mérete
    criterion='entropy',   # Homogenitási kritérium
    random_state=2025
)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
results['Decision Tree'] = dt_score

# B) Logisztikus Regresszió
lr = LogisticRegression(
    solver='liblinear',    # Solver beállítása
    max_iter=200,          # Iteráció beállítása
    random_state=2025
)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
results['Logistic Regression'] = lr_score

# C) Neurális Háló
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10), # Háló mérete (2 rejtett réteg, 10-10 neuron)
    activation='relu',           # Aktivációs függvény
    max_iter=500,
    random_state=2025
)
mlp.fit(X_train, y_train)
mlp_score = mlp.score(X_test, y_test)
results['Neural Network'] = mlp_score

print("5. Modellek pontossága (Accuracy):")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Legjobb modell kiválasztása
best_model_name = max(results, key=results.get)
if best_model_name == 'Decision Tree':
    best_model = dt
elif best_model_name == 'Logistic Regression':
    best_model = lr
else:
    best_model = mlp

print(f"\nA legjobb modell: {best_model_name}")
print("-" * 30)

# ---------------------------------------------------------
# 6. Tévesztési mátrix és ROC görbe (4 pont)
# ---------------------------------------------------------
# Tévesztési mátrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title(f'6. Feladat: Confusion Matrix ({best_model_name})')
plt.show()

# ROC Görbe (Csak bináris osztályozásnál egyszerű!)
if len(np.unique(y)) == 2:
    # Valószínűségek kellenek a pozitív osztályhoz (az 1-es indexű osztály)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'6. Feladat: ROC Görbe ({best_model_name})')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("ROC görbe kirajzolása csak bináris osztályozásnál támogatott ebben a kódban.")

# ---------------------------------------------------------
# 7. Nemfelügyelt tanítás (K-Means + PCA) (13 pont)
# ---------------------------------------------------------
print("7. Nemfelügyelt tanítás: K-Means optimalizálás...")

# Input attribútumok (X) használata
# Érdemes skálázni klaszterezés előtt
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db_scores = []
K_range = range(2, 51) # 2-től 50-ig

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=2025, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    # DB index: minél kisebb, annál jobb
    score = davies_bouldin_score(X_scaled, labels)
    db_scores.append(score)

# Optimális K megtalálása (ahol a DB index a legkisebb)
optimal_k_index = np.argmin(db_scores)
optimal_k = K_range[optimal_k_index]
print(f"Optimális klaszterszám (DB index alapján): {optimal_k}")
print(f"Legkisebb DB index: {db_scores[optimal_k_index]:.4f}")

# Vizualizáció DB index alakulására (opcionális, de hasznos)
plt.plot(K_range, db_scores, marker='o')
plt.xlabel('Klaszterszám (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('DB Index a klaszterszám függvényében')
plt.show()

# Végső klaszterezés az optimális K-val
final_kmeans = KMeans(n_clusters=optimal_k, random_state=2025, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

# PCA vizualizáció (2 dimenzió)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='viridis', edgecolor='k', s=50)
plt.title(f'7. Feladat: Klaszterek vizualizációja PCA-val (k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.show()