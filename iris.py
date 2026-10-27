import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df = pd.read_csv("iris.data", header=None, names=cols + ["classe"])
classes = df["classe"].unique()
cores = dict(zip(classes, ["tab:blue", "tab:orange", "tab:green"]))

# ---------- 1. Estatística descritiva ----------
print("== Frequência das classes ==")
print(df["classe"].value_counts(), "\n")

def resumo(d):
    return pd.DataFrame({"media": d[cols].mean(), "desvio": d[cols].std(),
                         "moda": d[cols].mode().iloc[0]})

print("== Geral ==")
print(resumo(df).round(3), "\n")
for c in classes:
    print("==", c, "==")
    print(resumo(df[df["classe"] == c]).round(3), "\n")

# ---------- 2. Relação comprimento x largura (sépala e pétala) ----------
print("== Correlação comprimento x largura ==")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, parte in enumerate(["sepal", "petal"]):
    for c in classes:
        d = df[df["classe"] == c]
        x, y = d[parte + "_length"], d[parte + "_width"]
        print(f"{parte:5s} {c:16s} r = {np.corrcoef(x, y)[0, 1]:.3f}")
        ax[i].scatter(x, y, label=c, color=cores[c], s=15)
    ax[i].set_xlabel(parte + " length (cm)")
    ax[i].set_ylabel(parte + " width (cm)")
    ax[i].set_title(parte)
ax[0].legend()
plt.tight_layout()
plt.savefig("fig_relacao.png", dpi=120)
plt.close()

# ---------- 3. Distribuição das classes em cada dimensão ----------
fig, ax = plt.subplots(2, 4, figsize=(14, 6))
for j, col in enumerate(cols):
    for c in classes:
        ax[0, j].hist(df[df["classe"] == c][col], alpha=0.6, label=c, color=cores[c])
    ax[0, j].set_title(col)
    ax[1, j].boxplot([df[df["classe"] == c][col] for c in classes], tick_labels=[c[5:] for c in classes])
ax[0, 0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig_distribuicao.png", dpi=120)
plt.close()

# ---------- 4. PCA ----------
X = df[cols].values
y = df["classe"].values
Xp = StandardScaler().fit_transform(X)
pca = PCA().fit(Xp)
Z = pca.transform(Xp)
var = pca.explained_variance_ratio_
print("\n== PCA ==")
print("Variância explicada:", var.round(3), " acumulada:", var.cumsum().round(3))
print("Componentes (linhas PC1..PC4):")
print(pd.DataFrame(pca.components_.T, index=cols, columns=["PC1", "PC2", "PC3", "PC4"]).round(3))
plt.figure(figsize=(5, 4))
for c in classes:
    m = y == c
    plt.scatter(Z[m, 0], Z[m, 1], label=c, color=cores[c], s=15)
plt.xlabel(f"PC1 ({var[0]:.1%})")
plt.ylabel(f"PC2 ({var[1]:.1%})")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig_pca.png", dpi=120)
plt.close()

# ---------- 5. Divisão 80/20 e K-NN ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n== K-NN (treino={len(Xtr)}, teste={len(Xte)}) ==")
scaler = StandardScaler().fit(Xtr)     # média/desvio só do treino
Xtr_n, Xte_n = scaler.transform(Xtr), scaler.transform(Xte)

acc = {}
for k in [1, 3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr_n, ytr)
    acc[k] = accuracy_score(yte, knn.predict(Xte_n))
    print(f"k={k:2d}  acurácia = {acc[k]:.3f}")

kb = max(acc, key=acc.get)
pred = KNeighborsClassifier(n_neighbors=kb).fit(Xtr_n, ytr).predict(Xte_n)
print(f"\nMelhor k = {kb}. Matriz de confusão (linha=real, coluna=predito):")
print(confusion_matrix(yte, pred, labels=classes))

# K-NN só com PC1 e PC2 (PCA ajustado no treino)
pca2 = PCA(n_components=2).fit(Xtr_n)
print("\nK-NN com PC1+PC2 apenas:")
for k in [1, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(pca2.transform(Xtr_n), ytr)
    print(f"k={k:2d}  acurácia = {accuracy_score(yte, knn.predict(pca2.transform(Xte_n))):.3f}")
