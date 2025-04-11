import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#You can change this to read in whichever implementation dataset you output.
df = pd.read_csv("data/output_mpi.csv")
features = df.drop("cluster", axis=1)
labels = df["cluster"]

pca = PCA(n_components=2)
proj = pca.fit_transform(features)

plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10')
plt.title("K-Means Cluster Visualization (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
#Cluster as last column
features = df.columns[:-1]
for i, component in enumerate(pca.components_[:2]):
    print(f"\nPC{i+1} contributions:")
    for feat, weight in zip(features, component):
        print(f"{feat:15s}: {weight:.3f}")

plt.colorbar()
plt.show()
