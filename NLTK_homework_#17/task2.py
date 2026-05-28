import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

df["tokens"] = df["clean_review"].apply(word_tokenize)

tokens_before = sum(len(tokens) for tokens in df["tokens"])

df["filtered_tokens"] = df["tokens"].apply(
    lambda tokens: [
        word for word in tokens
        if word not in stop_words and len(word) >= 3
    ]
)

tokens_after = sum(len(tokens) for tokens in df["filtered_tokens"])

print(f"Total tokens before cleaning: {tokens_before}")
print(f"Total tokens after cleaning: {tokens_after}")

print("\nExample filtered tokens:")
print(df["filtered_tokens"].head())