import chromadb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Step 1: Retrieve Data from ChromaDB ---

# Initialize the ChromaDB client (e.g., a persistent client)
# Make sure you have a collection with some documents and embeddings already added.
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("docs")

results = collection.get(
        ids=None, # By passing None or a list of all IDs, you retrieve all data
        include=["embeddings", "metadatas", "documents"]
    )

embeddings = results['embeddings']
metadatas = results['metadatas']

# Convert the list of lists to a NumPy array for scikit-learn
X = np.array(embeddings)
labels = [metadata['source'] for metadata in metadatas]

# --- Step 2: Dimensionality Reduction with scikit-learn ---

# Option 1: Principal Component Analysis (PCA)
# PCA is a linear method that is fast and works well for global structure.
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Option 2: t-Distributed Stochastic Neighbor Embedding (t-SNE)
# t-SNE is a non-linear method that is excellent for visualizing clusters.
# It's more computationally expensive than PCA but often provides better results for visualization.
print("Applying t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
X_tsne = tsne.fit_transform(X)

# --- Step 3: Visualization ---

def plot_embeddings(X_reduced, title):
    plt.figure(figsize=(8, 6))
    
    # Get unique labels and map them to colors
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(
            X_reduced[indices, 0],
            X_reduced[indices, 1],
            label=label,
            color=colors(i),
            s=50
        )
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize the results
plot_embeddings(X_pca, "PCA Visualization of ChromaDB Embeddings")
# plot_embeddings(X_tsne, "t-SNE Visualization of ChromaDB Embeddings")