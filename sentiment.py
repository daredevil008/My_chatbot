# sentiment.py
# (Content truncated for brevity in this environment demonstration)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()

def analyze_text(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.05:
        label = "Positive"
    elif compound < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"scores": scores, "compound": compound, "label": label}

def analyze_conversation(msgs):
    results = [analyze_text(m) for m in msgs]
    avg = sum(r["compound"] for r in results) / len(results)
    if avg > 0.05:
        label = "Positive"
    elif avg < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"per_message": results, "average_compound": avg, "final_label": label}
