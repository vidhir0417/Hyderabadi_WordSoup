# Text Mining Project - Solving the Hyderabadi Word Soup

This document focuses on analyzing restaurant reviews from Hyderabad, India, using various text mining techniques.

## Purpose

The primary objective of this project is to extract meaningful insights from a dataset of Zomato reviews for Hyderabad's restaurants. This involves:
* Analyzing restaurant characteristics and review metadata.
* Categorizing cuisine types using multilabel classification.
* Predicting restaurant ratings based on sentiment analysis.
* Identifying commonly mentioned dishes through co-occurrence analysis.
* Classifying reviews into emergent topics and interpreting their main subjects.

## Dataset

The project utilizes two main datasets:
1.  **Restaurants Dataset (105_restaurants.csv):** Includes 105 restaurants in Hyderabad, providing details like cost per person, collections, cuisine types and operating hours.
2.  **Reviews Dataset (10k_reviews.csv):** Contains metadata about reviewers and their feedback, including review text, rating, reviewer details, timestamp and number of images.

## Methodology

The project followed the CRISP-DM methodology, ensuring a structured and comprehensive approach to text mining.

### 1. Data Understanding & Exploration
* **Restaurants Dataset:** Identified most common cuisine types (North Indian, Chinese) and average dining cost.
* **Reviews Dataset:** Observed a dominance of extreme ratings (1, 4, 5) and a mean rating suggesting overrating. Most reviews are submitted late in the evening during summer.
* **Initial Preprocessing:** Dropped irrelevant columns (`Links`, `Timings`, metadata) and converted data types for efficiency.
* **Text Cleaning Pipeline:** Developed a flexible `preprocessing` function (`text_cleaner`) to handle lowercasing, punctuation/emoji removal, and other user-defined steps.
* **Feature Extraction (Reviews):** Applied Bag of Words (BoW) and TF-IDF (after splitting by rating) for feature extraction. Visualized common words and constructed a network graph of top co-occurring words.

### 2. Specific Data Preprocessing for Models

* **Multilabel Classification:** Merged restaurant and review data, binarized cuisine types, removed stopwords and lemmatized words.
* **Sentiment Analysis:** Retained punctuation and UTF-8 emojis as they convey sentiment. Stopwords were also kept to preserve sentence meaning.
* **Co-occurrence Analysis & Clustering:** Removed stopwords, punctuation and emojis. Lemmatized text and converted to lowercase.
* **Topic Modeling:** Used Bag of Words matrix and lemmatized tokens (converted to list format) for both Sklearn and Gensim libraries.

### 3. Modelling

* **Multilabel Classification:** Employed a classifier chain with decision trees, achieving an F1 score of 0.38 across over 40 different cuisine classes.
* **Sentiment Analysis:** Used VADER and TextBlob analyzers (full-text and per-sentence approaches).
    * Confirmed a fair correlation between review sentiment and ratings.
    * Achieved an $R^2$ of 0.5 (TextBlob full text) and a Mean Absolute Error (MAE) of 0.83, indicating reasonable prediction of ratings based on sentiment.
* **Co-occurrence & Clustering Analysis:**
    * Extracted unique food/drink nouns to construct a co-occurrence matrix and network graph of commonly mentioned dishes.
    * Attempted clustering (KMeans with BoW and Doc2Vec, HDBSCAN) to identify cuisine types based on review content. While clusters were formed, many were dominated by common cuisines (North Indian, Chinese).
* **Topic Modeling:** Performed Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) using both Sklearn and Gensim.
    * Explored 20, 15, 10 and 8 topics to find the optimal number.
    * LDA was found to be more interpretable due to coherent word distributions within topics.
    * Successfully classified reviews into emergent topics and extracted their main subjects, despite some word repetition across topics.

## Conclusion

The project successfully applied various text mining techniques to extract valuable insights from Hyderabad restaurant reviews. It demonstrated the ability to:
* Predict Zomato ratings based on sentiment (MAE of 0.83).
* Predict cuisine types using classifier chain models.
* Identify commonly co-occurring dishes.
* Segment reviews through clustering, though with some dominance by major cuisines.
* Classify reviews into interpretable topics.

While some challenges were faced (e.g., manual food word extraction, clustering dominance), the project provided significant insights into restaurant analysis from unstructured text data.

## References

* \[1\] N. V. Smirnov, A. S. Trifonov, "Classification of Incoming Messages of the University Admission Campaign", SmartIndustryCon, 2023.
* \[2\] S. Singh, T. Chauhan, V. Wahi, P. Meel, "Mining Tourists Opinions on Popular Indian Tourism Hotspots using Sentiment Analysis and Topic Modeling", ICCMC, 2021.
* \[3\] S. Liu, X. Fan, J. Chai, "A Clustering Analysis of News Text Based on Co-occurrence Matrix", IEEE, 2017.
