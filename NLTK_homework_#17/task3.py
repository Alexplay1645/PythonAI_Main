from collections import Counter
import matplotlib.pyplot as plt

all_tokens = []

for tokens in df["filtered_tokens"]:
    all_tokens.extend(tokens)

word_freq = Counter(all_tokens)

top_15 = word_freq.most_common(15)

print("Top 15 Frequent Words:\n")

for word, freq in top_15:
    print(f"{word}: {freq}")

words = [item[0] for item in top_15]
freqs = [item[1] for item in top_15]

plt.figure(figsize=(10, 6))
plt.barh(words, freqs)

plt.xlabel("Frequency")
plt.ylabel("Word")
plt.title("Top 15 Frequent Words")

plt.tight_layout()

plt.savefig("feedback_word_freq.png")

plt.show()

print("Chart saved as feedback_word_freq.png")