import pandas as pd
import re
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/raw/public_feedback_ar.csv")

# Simple Arabic text cleaning
def clean_text(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()
print(sentiment_counts)

plt.figure()
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution in Public Feedback")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Category vs sentiment
category_sentiment = pd.crosstab(df["category"], df["sentiment"])
print(category_sentiment)

category_sentiment.plot(kind="bar")
plt.title("Public Feedback by Category and Sentiment")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

print("\nINSIGHTS:")
print("- Most negative feedback is related to health and education.")
print("- Digital services show improvement but still receive complaints.")
