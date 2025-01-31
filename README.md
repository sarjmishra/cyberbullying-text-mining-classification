## Introduction
This project involves detecting cyberbullying from unstructured tweet data using text mining and machine learning techniques. The goal is to convert unstructured text data into structured numeric data, allowing the application of machine learning models to classify tweets as either cyberbullying or not. The following sections of this report outline the steps involved in building binary classifiers to detect cyberbullying.

## Dataset
The dataset contains 47,692 tweets, each of which is assigned a label in the "cyberbullying_type" column. The values in this column include "age", "ethnicity", "gender", "religion", "other_cyberbullying", and "not_cyberbullying". These values were mapped to two classes: "cyberbullying" (positive label) and "not_cyberbullying" (negative label) for binary classification.

## Text Cleaning
Text data cleaning involved several steps to remove irrelevant information and noise:
	•	Removed special characters like @ and #
	•	Removed hyperlinks, image/video links, and HTML tags
	•	Converted text to lowercase
	•	Removed plural forms and lemmatized parts of speech
	•	Removed stop words, punctuation, and numbers
The cleaning process used BeautifulSoup for HTML tag detection and regular expressions for removing unwanted text patterns. NLTK's WordNetLemmatizer was used for lemmatization. Any missing values created during the cleaning process were removed.

## Creation of Structured Numerical Dataset
The text data was transformed into a numerical dataset using the Term Frequency-Inverse Document Frequency (TF-IDF) approach. The dataset was represented as a document-term matrix (DTM), where each tweet (document) is represented as a set of unique tokens (terms). The TfidfVectorizer from sklearn was used to create the TF-IDF matrix.
	•	TF (Term Frequency): Measures how frequently a term appears in a document.
	•	IDF (Inverse Document Frequency): Measures the importance of a term across the entire corpus.
N-grams (unigrams, bigrams, and trigrams) were included in the representation to capture contextual meaning. The minimum document frequency (min_df) was set to 30, meaning terms must appear in at least 30 documents to be considered.

## Choice of Models
Two models were used for classification:
1. Support Vector Classifier (SVC)
2. Naive Bayes Classifier (NBC)
Both models were evaluated using the recall metric due to the class imbalance. Recall measures the percentage of actual cyberbullying tweets correctly identified by the model.

## Evaluation Metric
Given the class imbalance, Recall is the primary evaluation metric. The confusion matrix for this problem is:
	•	True Positive (TP): Correctly predicted cyberbullying tweets.
	•	False Positive (FP): Non-cyberbullying tweets incorrectly predicted as cyberbullying.
	•	True Negative (TN): Correctly predicted non-cyberbullying tweets.
	•	False Negative (FN): Cyberbullying tweets incorrectly predicted as non-cyberbullying.

The goal is to minimise false negatives, as it’s crucial to flag potential cyberbullying tweets.

## Model Performance
	•	SVC: Mean recall score = 85.05%
	•	NBC: Mean recall score = 82.45%
 
Both models performed well, with SVC slightly outperforming NBC. No overfitting was observed for either model.
•	SVC has a higher recall score, meaning it can identify more cyberbullying tweets compared to NBC. For example, SVC would correctly identify 31.87 million cyberbullying tweets out of 37.5 million, while NBC would identify 30.75 million on this dataset.
