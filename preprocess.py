import json
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

with open("sample.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if not text:  
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)

    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

df["processed_review"] = df["review_detail"].apply(preprocess_text)
df["processed_review"] = df["processed_review"].fillna("").astype(str)

preprocessed_data = df[["review_id", "processed_review"]].to_dict(orient="records")

with open("preprocessed_reviews.json", "w") as output_file:
    json.dump(preprocessed_data, output_file)

print("Preprocessed reviews saved to 'preprocessed_reviews.json'.") 