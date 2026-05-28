import pandas as pd
import re

reviews = [
    "The product quality is excellent and I am very satisfied.",
    "Customer service was helpful and resolved my issue quickly.",
    "I love this product, it works perfectly every day.",
    "The delivery was fast and the packaging was great.",
    "Good quality and affordable price. Highly recommended.",
    "The product stopped working after a week of use.",
    "Excellent design and very easy to use.",
    "Customer support was slow but eventually solved my problem.",
    "This is the best purchase I have made this year.",
    "The quality could be better for the price.",
    "Amazing experience and outstanding product performance.",
    "The item arrived damaged and customer service was disappointing.",
    "Very reliable and durable product.",
    "I am happy with the purchase and would buy again.",
    "The instructions were unclear and difficult to follow."
]

df = pd.DataFrame({"review": reviews})

df.to_csv("customer_feedback.csv", index=False)

print("CSV file created successfully.")

df = pd.read_csv("customer_feedback.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["clean_review"] = df["review"].apply(clean_text)

num_rows = len(df)

avg_length = df["clean_review"].apply(lambda x: len(x.split())).mean()

print(f"Number of reviews: {num_rows}")
print(f"Average review length: {avg_length:.2f} words")

print("\nCleaned Reviews:")
print(df[["clean_review"]].head())