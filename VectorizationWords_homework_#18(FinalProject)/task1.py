import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("reviews.csv")

print("Кількість записів:", len(df))
print(df.head())

stop_words = set(stopwords.words('english'))

tokenized_sentences = []

for text in df['review'].dropna():

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokenized_sentences.append(tokens)

print("\nКількість речень:", len(tokenized_sentences))

print("\nПриклад токенізованого речення:")
print(tokenized_sentences[0])