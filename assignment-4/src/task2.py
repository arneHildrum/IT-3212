import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('../data/Tetuan_City_power_consumption.csv')


# Parse datetime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Extract time features
df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month
df["Weekday"] = df["DateTime"].dt.weekday

# Drop original datetime
df = df.drop(columns=["DateTime"])

# Handle missing values
df = df.interpolate()

# --------------------------
# 2. FEATURE ENGINEERING
# --------------------------

# Interaction features
df["Power_Sum"] = df[["Zone 1 Power Consumption",
                      "Zone 2  Power Consumption",
                      "Zone 3  Power Consumption"]].sum(axis=1)

df["Power_Std"] = df[["Zone 1 Power Consumption",
                      "Zone 2  Power Consumption",
                      "Zone 3  Power Consumption"]].std(axis=1)

df["Power_Ratio12"] = df["Zone 1 Power Consumption"] / (
                       df["Zone 2  Power Consumption"] + 1e-6)

# --------------------------
# 3. FEATURE SELECTION
# --------------------------

# Remove near-zero variance features
vt = df.loc[:, df.var() > 1e-6]

# Remove highly correlated features ( > 0.9 )
corr = vt.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
df_final = vt.drop(columns=to_drop)

# --------------------------
# SCALE DATA
# --------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df_final)

# --------------------------
# 4. CLUSTERING ALGORITHMS
# --------------------------

# ---- K-Means ----
kmeans = KMeans(n_clusters=3, random_state=0)
labels_kmeans = kmeans.fit_predict(X)

# ---- Gaussian Mixture ----
gmm = GaussianMixture(n_components=3, random_state=0)
labels_gmm = gmm.fit_predict(X)

# ---- Hierarchical Clustering (Ward) ----
hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hier = hier.fit_predict(X)
# --------------------------
# 5. EVALUATION METRICS
# --------------------------


def evaluate(X, labels, name):
    if len(set(labels)) > 1 and -1 not in set(labels):
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        sil = np.nan
        ch = np.nan

    print(f"{name}:")
    print(f"  Silhouette Score: {sil}")
    print(f"  Calinski-Harabasz Score: {ch}")
    print(f"  Unique Clusters: {set(labels)}\n")


evaluate(X, labels_kmeans, "K-Means")
evaluate(X, labels_gmm, "GMM")
evaluate(X, labels_hier, "Hierarchical Clustering")


# --------------------------
# 6. VISUALIZATION (PCA 2D)
# --------------------------


# Reduce data to 2D for visualization
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

def plot_clusters(X2, labels, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()

# Plot for each algorithm
plot_clusters(X_2D, labels_kmeans, "K-Means Clusters (PCA 2D)")
plot_clusters(X_2D, labels_gmm, "GMM Clusters (PCA 2D)")
plot_clusters(X_2D, labels_hier, "Hierarchical Clusters (PCA 2D)")
