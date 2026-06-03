from gensim.models import Word2Vec
import pandas as pd

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1
)

keywords = [
    "good",
    "service",
    "price",
    "delivery",
    "quality",
    "product",
    "customer",
    "fast",
    "excellent",
    "support"
]

results = []

for word in keywords:

    if word in model.wv:

        similar_words = model.wv.most_similar(word, topn=5)

        close_words = [w for w, s in similar_words]
        similarities = [round(s, 3) for w, s in similar_words]

        results.append([
            word,
            ", ".join(close_words),
            ", ".join(map(str, similarities))
        ])

result_df = pd.DataFrame(
    results,
    columns=[
        "Ключове слово",
        "Близькі слова",
        "Косинусна схожість"
    ]
)

print(result_df)

result_df.to_csv(
    "word_similarity_results.csv",
    index=False
)