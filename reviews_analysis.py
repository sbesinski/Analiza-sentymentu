import json
import pandas as pd
from transformers import pipeline
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor



# Step 1: Load preprocessed data from JSON
print("Loading preprocessed data from JSON...")
with open("preprocessed_reviews.json", "r") as file:
    preprocessed_data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(preprocessed_data)
print(f"Loaded {len(df)} reviews from the JSON file.")

# Step 2: Save preprocessed DataFrame to Parquet for efficiency
parquet_file = "preprocessed_reviews.parquet"
df.to_parquet(parquet_file, index=False)
print(f"Preprocessed data saved to '{parquet_file}'.")

# Step 3: Load data from Parquet file
print("Loading data from Parquet file...")
df = pd.read_parquet(parquet_file)
print(f"Data loaded from '{parquet_file}', {len(df)} records found.")

# Step 4: Initialize Sentiment and Emotion Analysis Pipelines
print("Initializing sentiment and emotion analysis pipelines...")

# Check for MPS (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Sentiment analysis pipeline
sentiment_analyzer = pipeline(
    'sentiment-analysis', 
    model="nlptown/bert-base-multilingual-uncased-sentiment", 
    device=0 if device != torch.device("cpu") else -1
)

# Emotion analysis pipeline
emotion_analyzer = pipeline(
    'text-classification', 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=None, 
    device=0 if device != torch.device("cpu") else -1
)

# Step 5: Function to analyze reviews in batches
def analyze_batch(reviews):
    results = []
    for text in reviews:
        try:
            text = text[:512]  # Truncate to 512 tokens
            sentiment = sentiment_analyzer(text)[0]
            emotion_scores = emotion_analyzer(text)[0]
            emotion = max(emotion_scores, key=lambda x: x['score'])  # Most likely emotion

            results.append({
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "emotion_label": emotion["label"],
                "emotion_score": emotion["score"]
            })
        except Exception as e:
            print(f"Error analyzing review: {e}")
            results.append({})
    return results

# Step 6: Perform sentiment and emotion analysis with threading
print("Performing sentiment and emotion analysis on the reviews...")

batch_size = 100
batches = [df["processed_review"][i:i + batch_size] for i in range(0, len(df), batch_size)]

# Use ThreadPoolExecutor to parallelize batch processing
analysis_results = []
total_batches = len(batches)

with ThreadPoolExecutor() as executor:
    for batch_number, batch_result in enumerate(executor.map(analyze_batch, batches), start=1):
        analysis_results.extend(batch_result)
        print(f"Processed batch {batch_number}/{total_batches} ({batch_size * batch_number}/{len(df)} reviews processed).")

# Add analysis results to the DataFrame
df["analysis"] = analysis_results
df["sentiment"] = df["analysis"].apply(lambda x: x.get("sentiment_label", ""))
df["sentiment_confidence"] = df["analysis"].apply(lambda x: x.get("sentiment_score", 0))
df["emotion"] = df["analysis"].apply(lambda x: x.get("emotion_label", ""))
df["emotion_confidence"] = df["analysis"].apply(lambda x: x.get("emotion_score", 0))

# Step 7: Save results to CSV
results_file = "sentiment_emotion_analysis_results.csv"
df.to_csv(results_file, index=False)
print(f"Sentiment and emotion analysis completed. Results saved to '{results_file}'.")

# Step 8: Generate sentiment distribution plot
print("Generating sentiment distribution plot...")

# Count the number of reviews for each sentiment
sentiment_counts = Counter(df["sentiment"])

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color='skyblue')

# Add titles and labels
plt.title("Sentiment Analysis Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=14)
plt.ylabel("Number of Reviews", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot to a PNG file
plot_file = "sentiment_analysis_distribution.png"
plt.savefig(plot_file, format='png', dpi=300)
print(f"Sentiment distribution plot saved to '{plot_file}'.")

# Show the plot (optional)
plt.show()