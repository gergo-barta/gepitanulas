"""
Feladatok

    Olvassa be a banknote_authentication.txt adatállományt a https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/ URL címről és írassa ki a legfontosabb jellemzőit, úm. rekordok száma, attribútumok száma és osztályok száma. (3 pont)
    Csináljon DataFrame-t az adatokból és ábrázolja őket parallel_coordinates ábrával ahol színezze a két osztálytt kékkel és pirossal. (3 pont)
    Partícionálja az adatállományt 70% tanító és 30% tesztállományra. Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (2 pont)
    Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: döntési fa (6 mélység, Gini homogenitási kritérium), logisztikus regresszió (alapbeállítással) és Gauss-féle naív Bayes. A teszt pontosság alapján hasonlítsa össze az illesztett modelleket, nyomtassa ki a legjobb teszt pontosságát. (10 pont)
    Számolja ki a 4. pont legjobb modelljére a teszt tévesztési mátrixot, amelyet jelenítsen is meg egyben. (4 pont)
    Rajzolja ki a 4. pont legjobb modelljének ROC görbéjét. (4 pont)
    Végezzen nemfelügyelt tanítást a K-közép módszerrel a tanító állomány input attribútumain. Határozza meg az optimális klaszterszámot 30-ig való kereséssel a DB index alapján a teszt állományon. (7 pont)
    K=2 klaszterszám mellett vizualizálja a klasztereket egy pontdiagramon, ahol a két koordináta egy 2 dimenziós PCA eredménye. (7 pont)

Összesen: 40 pont
 
A megoldásokat egy python fájlban, az egyes pontokat kommentelve (#1. feladat, ...) töltse fel. A fájlokat nevezze el az alábbi konvenció alapján: Vezetéknév_Keresztnév_Neptunkód.py

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. FELADAT: Adatbeolvasás és jellemzők
# ---------------------------------------------------------
print("--- 1. Feladat ---")
url = "https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/banknote_authentication.txt"

# Mivel .txt és általában nincs fejléc, header=None-t használunk
df = pd.read_csv(url, header=None)

# Oszlopnevek hozzáadása a könnyebb kezelhetőségért (szabványos banknote dataset nevek)
# Az utolsó oszlop az osztály (class)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

rec_num = df.shape[0]
attr_num = df.shape[1] - 1 # Mínusz a célváltozó
class_num = df['class'].nunique()

print(f"Rekordok száma: {rec_num}")
print(f"Attribútumok száma: {attr_num}")
print(f"Osztályok száma: {class_num}")

# ---------------------------------------------------------
# 2. FELADAT: DataFrame és Parallel Coordinates (Kék/Piros)
# ---------------------------------------------------------
print("\n--- 2. Feladat: Parallel Coordinates ---")
# Megjegyzés: A parallel_coordinates ábrázolásnál a 'class' oszlop alapján színezünk.
# A feladat kék és piros színeket kért.

plt.figure(figsize=(10, 6))
parallel_coordinates(df, 'class', color=['blue', 'red'])
plt.title("2. Feladat: Parallel Coordinates (Kék/Piros)")
plt.xlabel("Attribútumok")
plt.ylabel("Értékek")
plt.show()

# ---------------------------------------------------------
# 3. FELADAT: Particionálás (70/30) és keverés
# ---------------------------------------------------------
print("\n--- 3. Feladat: Particionálás ---")

X = df.drop('class', axis=1)
y = df['class']

# random_state = 2025 (idei év), shuffle=True (keverés)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=2025
)

print(f"Tanító halmaz mérete: {X_train.shape}")
print(f"Teszt halmaz mérete: {X_test.shape}")

# ---------------------------------------------------------
# 4. FELADAT: Felügyelt tanítás és összehasonlítás
# ---------------------------------------------------------
print("\n--- 4. Feladat: Modellek ---")

models = {}
results = {}

# a) Döntési fa (6 mélység, Gini)
dt = DecisionTreeClassifier(max_depth=6, criterion='gini', random_state=2025)
dt.fit(X_train, y_train)
models['Decision Tree'] = dt
results['Decision Tree'] = accuracy_score(y_test, dt.predict(X_test))

# b) Logisztikus regresszió (alapbeállítás)
lr = LogisticRegression(random_state=2025)
lr.fit(X_train, y_train)
models['Logistic Regression'] = lr
results['Logistic Regression'] = accuracy_score(y_test, lr.predict(X_test))

# c) Gauss-féle Naív Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
models['Naive Bayes'] = gnb
results['Naive Bayes'] = accuracy_score(y_test, gnb.predict(X_test))

# Eredmények kiírása
for name, acc in results.items():
    print(f"{name} pontossága: {acc:.4f}")

# Legjobb modell kiválasztása
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n=> A legjobb modell: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

# ---------------------------------------------------------
# 5. FELADAT: Tévesztési mátrix (Legjobb modell)
# ---------------------------------------------------------
print(f"\n--- 5. Feladat: Tévesztési mátrix ({best_model_name}) ---")

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title(f"5. Feladat: Confusion Matrix ({best_model_name})")
plt.show()

# ---------------------------------------------------------
# 6. FELADAT: ROC Görbe (Legjobb modell)
# ---------------------------------------------------------
print(f"\n--- 6. Feladat: ROC görbe ({best_model_name}) ---")

# Valószínűségek kellenek a pozitív osztályhoz (ami általában az 1-es)
# Egyes modellek (pl. SVM) nem mindig adnak vissza valószínűséget alapból, 
# de a fentiek (DT, LR, NB) támogatják a predict_proba-t.
y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'6. Feladat: ROC Görbe ({best_model_name})')
plt.legend(loc="lower right")
plt.show()

# ---------------------------------------------------------
# 7. FELADAT: K-Means optimalizálás (Tanítás: Train, DB index: Test)
# ---------------------------------------------------------
print("\n--- 7. Feladat: K-Means optimalizálás (DB index) ---")

# Skálázás (klaszterezésnél ajánlott, bár a feladat nem írta elő kötelezően,
# de a "helyes" megoldáshoz általában hozzátartozik).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

db_scores = []
# Keresés 30-ig (tehát 2-től 30-ig fut a ciklus)
K_range = range(2, 31)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=2025, n_init=10)
    
    # Tanítás a TANÍTÓ (Train) állományon
    kmeans.fit(X_train_scaled)
    
    # DB index számítása a TESZT (Test) állományon (a feladat specifikus kérése)
    # Ehhez meg kell jósolni a teszt adatok klasztereit
    labels_test = kmeans.predict(X_test_scaled)
    
    score = davies_bouldin_score(X_test_scaled, labels_test)
    db_scores.append(score)

# Optimális K keresése (Minimum DB index)
optimal_k_idx = np.argmin(db_scores)
optimal_k = K_range[optimal_k_idx]

print(f"Optimális klaszterszám: {optimal_k}")
print(f"Legjobb DB index érték: {db_scores[optimal_k_idx]:.4f}")

# ---------------------------------------------------------
# 8. FELADAT: PCA vizualizáció (K=2)
# ---------------------------------------------------------
print("\n--- 8. Feladat: K=2 PCA Vizualizáció ---")

# Fix K=2 klaszterezés (ahogy a feladat kéri)
final_kmeans = KMeans(n_clusters=2, random_state=2025, n_init=10)
final_kmeans.fit(X_train_scaled)
# Vizualizálni érdemes a teljes vagy a teszt adatot, itt most a Train-t rajzoljuk ki
# (vagy következetesen azt, amit klasztereztünk).
labels_viz = final_kmeans.labels_ 

# PCA 2 dimenzióra
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_viz, cmap='viridis', s=50, alpha=0.7)
plt.title("8. Feladat: K=2 Klaszterek PCA (2D) vetületen")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()
