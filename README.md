{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Bold;\f1\froman\fcharset0 Times-Roman;\f2\fmodern\fcharset0 Courier;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid201\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid3}
{\list\listtemplateid4\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid301\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid4}
{\list\listtemplateid5\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid401\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid5}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}{\listoverride\listid4\listoverridecount0\ls4}{\listoverride\listid5\listoverridecount0\ls5}}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh18000\viewkind0
\deftab720
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Introduction\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 This project involves detecting cyberbullying from unstructured tweet data using text mining and machine learning techniques. The goal is to convert unstructured text data into structured numeric data, allowing the application of machine learning models to classify tweets as either cyberbullying or not. The following sections of this report outline the steps involved in building binary classifiers to detect cyberbullying.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Dataset\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The dataset contains 47,692 tweets, each of which is assigned a label in the "cyberbullying_type" column. The values in this column include "age", "ethnicity", "gender", "religion", "other_cyberbullying", and "not_cyberbullying". These values were mapped to two classes: "cyberbullying" (positive label) and "not_cyberbullying" (negative label) for binary classification.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Text Cleaning\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Text data cleaning involved several steps to remove irrelevant information and noise:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Removed special characters like 
\f2\fs26 @
\f1\fs24  and 
\f2\fs26 #
\f1\fs24 \
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Removed hyperlinks, image/video links, and HTML tags\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Converted text to lowercase\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Removed plural forms and lemmatized parts of speech\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Removed stop words, punctuation, and numbers\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 The cleaning process used BeautifulSoup for HTML tag detection and regular expressions for removing unwanted text patterns. NLTK's WordNetLemmatizer was used for lemmatization. Any missing values created during the cleaning process were removed.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Creation of Structured Numerical Dataset\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The text data was transformed into a numerical dataset using the Term Frequency-Inverse Document Frequency (TF-IDF) approach. The dataset was represented as a document-term matrix (DTM), where each tweet (document) is represented as a set of unique tokens (terms). The 
\f2\fs26 TfidfVectorizer
\f1\fs24  from sklearn was used to create the TF-IDF matrix.\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0
\f0\b \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 TF (Term Frequency)
\f1\b0 : Measures how frequently a term appears in a document.\
\ls2\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 IDF (Inverse Document Frequency)
\f1\b0 : Measures the importance of a term across the entire corpus.\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 N-grams (unigrams, bigrams, and trigrams) were included in the representation to capture contextual meaning. The minimum document frequency (
\f2\fs26 min_df
\f1\fs24 ) was set to 30, meaning terms must appear in at least 30 documents to be considered.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Choice of Models\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Two models were used for classification:\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 1. Support Vector Classifier (SVC)
\f1\b0\fs24 \

\f0\b\fs28 2. Naive Bayes Classifier (NBC)
\f1\b0\fs24 \
\pard\pardeftab720\sa240\partightenfactor0
\cf0 Both models were evaluated using the recall metric due to the class imbalance. Recall measures the percentage of actual cyberbullying tweets correctly identified by the model.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Evaluation Metric\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Given the class imbalance, 
\f0\b Recall
\f1\b0  is the primary evaluation metric. The confusion matrix for this problem is:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls3\ilvl0
\f0\b \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 True Positive (TP)
\f1\b0 : Correctly predicted cyberbullying tweets.\
\ls3\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 False Positive (FP)
\f1\b0 : Non-cyberbullying tweets incorrectly predicted as cyberbullying.\
\ls3\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 True Negative (TN)
\f1\b0 : Correctly predicted non-cyberbullying tweets.\
\ls3\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 False Negative (FN)
\f1\b0 : Cyberbullying tweets incorrectly predicted as non-cyberbullying.\
\pard\tx566\pardeftab720\partightenfactor0
\cf0 \
\pard\pardeftab720\sa240\partightenfactor0
\cf0 The goal is to minimise false negatives, as it\'92s crucial to flag potential cyberbullying tweets.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Model Performance\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls4\ilvl0
\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 SVC
\f1\b0 : Mean recall score = 85.05%\
\ls4\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 NBC
\f1\b0 : Mean recall score = 82.45%\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 Both models performed well, with SVC slightly outperforming NBC. No overfitting was observed for either model.\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls5\ilvl0
\f0\b \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 SVC
\f1\b0  has a higher recall score, meaning it can identify more cyberbullying tweets compared to NBC. For example, SVC would correctly identify 31.87 million cyberbullying tweets out of 37.5 million, while NBC would identify 30.75 million on this dataset.}