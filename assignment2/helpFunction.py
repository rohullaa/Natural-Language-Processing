import nltk
import random
from nltk.corpus import movie_reviews
import numpy as np
import scipy as sp
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

raw_movie_docs = [(movie_reviews.raw(fileid), category) for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]

random.seed(2920)
random.shuffle(raw_movie_docs)

def tokenize_and_embedding(text_data):
    data = []

    for i in range(len(text_data)):
        embeddings = []
        token_data = nltk.word_tokenize(' '.join([str(w) for w in text_data[i][0]]))
        for w in token_data:
            if w in wv:
                embeddings.append(wv[w])
        data.append([embeddings,text_data[i][1]])
    return data

def split_target_text(text_data):
    target = []
    texts = []
    for doc in text_data:
        texts.append(doc[0])
        target.append(doc[1])

    return target,texts

def find_mean(train_texts):
    list_of_means = []
    for i in range(len(train_texts)):
        text = train_texts[i]
        text = np.array(text)
        mean_of_text = []
        for j in range(len(text[0])):
            mean_of_text.append(np.mean(text[:,j]))
        list_of_means.append(mean_of_text)
    return list_of_means

raw_movie_docs = tokenize_and_embedding(raw_movie_docs)

#spitting the dataset
movie_test = raw_movie_docs[:200]
movie_dev = raw_movie_docs[200:]
train_data = movie_dev[:1600]
dev_test_data = movie_dev[1600:]

#splitting the target and the text
train_target, train_texts = split_target_text(train_data)
dev_test_target, dev_test_texts = split_target_text(dev_test_data)

#computing the mean for each document
train_texts = find_mean(train_texts)
dev_test_texts = find_mean(dev_test_texts)


#run the classifier
def tunning_logistic():
    print("Running logistic regression classifier: \n")
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,2000.0,3000.0]
    for C in C_values:
        log_reg = LogisticRegression(solver='liblinear',C=C)
        log_reg.fit(train_texts,train_target)
        acc = log_reg.score(dev_test_texts,dev_test_target)
        print(f'C = {C:7.2f} - accuracy = {acc:4.4f}')

tunning_logistic()
