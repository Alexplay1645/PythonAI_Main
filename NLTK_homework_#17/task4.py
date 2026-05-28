from turtle import pd
from nltk import bigrams

all_bigrams = []

for tokens in df["filtered_tokens"]:
    all_bigrams.extend(list(bigrams(tokens)))

bigram_freq = Counter(all_bigrams)

top_10_bigrams = bigram_freq.most_common(10)

bigram_df = pd.DataFrame(
    [
        (" ".join(bigram), freq)
        for bigram, freq in top_10_bigrams
    ],
    columns=["Bigram", "Frequency"]
)

print("Top 10 Bigrams:\n")
print(bigram_df)

bigram_df.to_csv("feedback_bigrams.csv", index=False)

print("\nResults saved as feedback_bigrams.csv")