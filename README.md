IMDb Movie Sentiment Analysis

Project Overview:
- Understanding customer feedback is crucial for businesses. This project uses Natural Language Processing (NLP) to automatically classify movie reviews as Positive or Negative.

The goal of the project was to build a model that can read raw, messy English text and determine the underlying sentiment with high accuracy.

The Data:
* Source: [IMDb Dataset (50k Reviews)](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Size: 50,000 rows (Balanced: 25k Positive, 25k Negative).
* Input: Raw text review (e.g., "I loved this movie!").
* Output: Binary Sentiment (Positive/Negative).

Results:
* Accuracy: 84.77%
* Key Insight: The model excels at identifying strong sentiment words ("Amazing", "Terrible") but can struggle with sarcasm (e.g., "The popcorn was the best part").

How to Run
1.  Clone the repo.
2.  Install dependencies: `pip install pandas scikit-learn seaborn`.
3.  Run the script: `python imdb.py`.

Key Challenges & Solutions:
- The Problem: Initial visualizations showed that the model was relying on generic nouns like "movie" and "film" as top predictors, which is logically incorrect (both good and bad reviews use these words).

The Solution:
* Analyzed the `feature_log_prob_` of the Naïve Bayes model.
* Identified "noise" words common to the movie domain.
* Extended the standard English stop-word list to include: `['movie', 'film', 'just', 'watch']`.

The Result: The model's top features shifted to strong sentiment indicators:
* Negative: "Waste", "Worst", "Awful".
* Positive: "Perfect", "Excellent", "Incredible".

Author:
* Chukwuemeka Eugene Obiyo
* LinkedIn: https://www.linkedin.com/in/chukwuemekao/
* Email: praise609@gmail.com
