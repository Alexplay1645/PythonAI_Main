from pyexpat import model
from sklearn.cluster import KMeans
import pandas as pd

words = list(model.wv.index_to_key)

word_vectors = [model.wv[word] for word in words]

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(word_vectors)

cluster_df = pd.DataFrame({
    "word": words,
    "cluster": clusters
})

for cluster_num in range(5):

    print(f"\nКЛАСТЕР {cluster_num}")

    cluster_words = cluster_df[
        cluster_df["cluster"] == cluster_num
    ]["word"].tolist()

    print(cluster_words[:10])