import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
print("Downloading dataset...") 

imdb = pd.read_csv(url)

print(f"Dataset Shape: {imdb.shape}")
print(imdb.head())

#checking balance 
sns.countplot(x='sentiment', data=imdb)
plt.show() 

import re
from sklearn.model_selection import train_test_split

#defining the cleaning function
def clean_text(text):
    #removing html tags
    text = re.sub(r'<.*?>', '', text)
    #removing non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    #converting to lowercase
    text = text.lower()
    return text

#applying chnages to dataset
print("Cleaning text...")
imdb['cleaned_review'] = imdb['review'].apply(clean_text)

#checking the difference
print("\n--- Original ---")
print(imdb['review'].iloc[0][:100])
print("\n--- Cleaned ---")
print(imdb['cleaned_review'].iloc[0][:100])

#splitting into train/test
#using standard 80/20 split
x = imdb['cleaned_review']
y = imdb['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(x_train)} reviews, Testing on {len(x_test)} reviews.")

#vectorization (tf-idf) 
from sklearn.feature_extraction.text import TfidfVectorizer
#intializing the vectorizer
#max_features=5000 (this prevents my machine from crashing [extend to 50000 if you have an excellent machine])
#stop_words='english': removing words like 'the, 'is', at'
#adding domain-specific words that are confusing thr model
imdb_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['movie', 'film', "don't", ' just', 'did', 'make']
tfidf = TfidfVectorizer(max_features=5000, stop_words=imdb_stop_words)
print("Vectorizing data...") 

#fit and transform
#learning from training data only, then apply to both
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test) 
print(f"Shape of Training Matrix: {x_train_tfidf.shape}")
print(f"Shape of Testing Matrix: {x_test_tfidf.shape}") 

#using naive bayes algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

#initialization and training
nb_model = MultinomialNB()
nb_model.fit(x_train_tfidf, y_train) 

#prediction on test set
y_pred = nb_model.predict(x_test_tfidf)

#evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))

def predict_sentiment(raw_review):
    cleaned = clean_text(raw_review)
    #vectorize (rransform only! do NOT fit!)
    vectorized = tfidf.transform([cleaned])
    #prediction
    prediction = nb_model.predict(vectorized)[0]
    #getting probability
    proba = nb_model.predict_proba(vectorized).max() * 100

    return f"Verdict: {prediction.upper()} ({proba:.2}% confident)"

#running tests
review1 = "I absolutely loved this film!"
review2 = "Horrible Movie!"
review3 = "It was okay, i guess! Could be better!"

print(f"Review 1: {predict_sentiment(review1)}")
print(f"Review 2: {predict_sentiment(review2)}")
print(f"Review 3: {predict_sentiment(review3)}")

from sklearn.metrics import confusion_matrix
#adding confusion matrix visuals
cm_imdb = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
cm_percent = cm_imdb.astype('float') / cm_imdb.sum()

plt.figure(figsize=(6,5))
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Greens', cbar=False,
            xticklabels=['Predicted Pos', 'Predicted Neg'],
            yticklabels=['Actual Pos', 'Actual Neg'])
plt.title('Confusion Matrix: Sentiment Analysis')
plt.show() 

#2nd visuals: what drives sentiments
#extracting log probabilities from the Naive Bayes model
feature_names = tfidf.get_feature_names_out()
neg_prob = nb_model.feature_log_prob_[0,:]
pos_prob = nb_model.feature_log_prob_[1,:] 

#organising into dataframe
feature_imdb = pd.DataFrame({
    'Word': feature_names,
    'Negative_LogProb': neg_prob,
    'Positive_LogProb': pos_prob
}) 

#getting top 10 for each side
top_pos = feature_imdb.sort_values(by='Positive_LogProb', ascending=False).head(10)
top_neg = feature_imdb.sort_values(by='Negative_LogProb', ascending=False).head(10)

fig, axes = plt.subplots(1,2, figsize=(14,6))
#positive plot
sns.barplot(x='Positive_LogProb', y='Word', data=top_pos, ax=axes[0], palette='Greens_r')
axes[0].set_title('Top 10 words in Positive Reviews')
axes[0].set_xlabel('Log Probability (Higher is better)')
#negative plot
sns.barplot(x='Negative_LogProb', y='Word', data=top_neg, ax=axes[1], palette='Reds_r')
axes[1].set_xlabel('Log Probability (Higher is better)')
axes[1].set_title('Top 10 Words in NEGATIVE Reviews')
plt.tight_layout() 
plt.show() 

