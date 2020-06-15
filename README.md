# SMS-Spam-Collectioin

# Prerequisites
- Pandas
- Matplotlib
- Seaborn
- Nltk
- Scikit Learn
- Regular Expressions

# Introduction

The SMS Spam Collection v.1 is a public set of SMS labeled messages that have been collected for mobile phone spam research. It has one collection composed by 5,574 English, real and non-enconded messages, tagged according being legitimate (ham) or spam.

Some questions arise when we take a look at the data set are:

Is there any pattern that could help us classify each message as ham or spam at a first view?
Which are the most common and most significant words for ham and spam messages?
How well our model predicts a new message as ham or spam?

# Exploratory Data Analysis

Our model was totally unbalanced and thus our final predictions could be changed if we had worked on a different dataset more balanced.

Eda uncover some interesting characteristics. We could predict if a message could be a spam or not by compute the length of its text. This is not an accurate way to do work but it could be a first check.

# Data Visualization

Seaborn and matplotlib once again were the tools we work with to do our visualizations.

# Natural Languange Process

**Bag of words model**

Machine learning models needs to ingest data in a structured form, a matrix where the rows represents observations and the columns are features/attributes. When working with text data, we need a method to convert this unstructured data into a form that the machine learning model can work with. One technique to transform text data into a matrix is to count the number of appearances of each word in each document. This technique is called the bag of words model. The model gets its name because each document is viewed as a bag holding all the words, disregarding word order, context, and grammar. After applying the bag of words model to a corpus, the resulting matrix will exhibit patterns that a machine learning model can exploit.

**The CountVectorizer transformer**

The bag of words model is found in scikit-learn with the CountVectorizer transformer. Note, scikit-learn uses the word Vectorizer to refer to transformers that convert a data structure (like a dictionary) into a NumPy array. Since it is a transformer, we need to first fit the object and then call transform.

**Stemming and lemmatization**

Stemming is the process of reducing a word to its stem. Note, the stemming process is not 100% effective and sometimes the resulting stem is not an actual word. For example, the popular Porter stemming algorithm applied to "argues" and "arguing" returns "argu".

Lemmatization is the process of reducing a word to its lemma, or the dictionary form of the word. It is a more sophisticated process than stemming as it considers context and part of speech. Further, the resulting lemma is an actual word.

## Term frequency-inverse document frequency

The `CountVectorizer` creates a feature matrix of raw counts. Using raw counts has two problems, documents vary widely in length and the counts will be large for common words such as "the" and "is". We need to use a weighting scheme that considers the aforementioned attributes. The term frequency-inverse document frequency, **tf-idf** for short, is a popular weighting scheme to improve the simple count based data from the bag of words model. It is the product of two values, the term frequency and the inverse document frequency. There are several variants but the most popular is defined below.


# Conclusions

1. We uncover hidden patterns by using the length of the messages. It is not a good metric of course but it could be a first check to recognize a spam or ham message.

2. We find the most common words for each category. But as long as most common doesn't mean the most significant we also find the most significants words for each category.

3. We trained two models. A RandomForestClassifier and a SGDClassifier. Both did very good with predictions but SGD did a bit better.

  We also tuned their parameters and we successfully improved their accuracies. One way we could improve our model it might be to create   a function to extact all the verbs for each message and using feature union to combine the results table and work on them.
