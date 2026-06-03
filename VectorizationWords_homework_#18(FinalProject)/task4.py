import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

vectors = np.array(word_vectors)

pca = PCA(n_components=2)

reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(14, 10))

scatter = plt.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1],
    c=clusters
)

for i, word in enumerate(words[:100]):

    plt.annotate(
        word,
        (
            reduced_vectors[i, 0],
            reduced_vectors[i, 1]
        ),
        fontsize=8
    )

plt.title("Semantic Word Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.legend(
    *scatter.legend_elements(),
    title="Clusters"
)

plt.savefig(
    "semantic_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()