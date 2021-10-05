# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # IN4080 – Natural Language Processing
# 
# This assignment has two parts:
# * Part A. Sequence labeling
# * Part B. Word embeddings
# %% [markdown]
# ## Part A
# 
# In this part we will experiment with sequence classification and tagging. We will combine some of
# the tools for tagging from NLTK with scikit-learn to build various taggers.We will start with simple
# examples from NLTK where the tagger only considers the token to be tagged—not its context—
# and work towards more advanced logistic regression taggers (also called maximum entropy taggers).
# Finally, we will compare to some tagging algorithms installed in NLTK.

# %%
import re
import pprint
import nltk
from nltk.corpus import brown
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]


# %%
def pos_features(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features

class ConsecutivePosTagger(nltk.TaggerI):
    def __init__(self, train_sents, features=pos_features):
        self.features = features
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = self.features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


# %%
tagger = ConsecutivePosTagger(train_sents)
print(round(tagger.evaluate(test_sents), 4))

# %% [markdown]
# ### 1) Tag set and baseline
# 
# **Part a:** Tag set and experimental set-up

# %%
def split_data(tagged_sents_uni):
    slice_ind = round(size*10/100)
    news_test = tagged_sents_uni[:slice_ind]
    news_dev_test = tagged_sents_uni[slice_ind:slice_ind*2]
    news_train = tagged_sents_uni[slice_ind*2:]

    return news_test, news_dev_test,news_train


# %%
tagged_sents_uni = brown.tagged_sents(categories='news',tagset = 'universal')
news_test, news_dev_test,news_train = split_data(tagged_sents_uni)


# %%
tagger_a = ConsecutivePosTagger(news_train)
print(round(tagger_a.evaluate(news_dev_test), 4))

# %% [markdown]
# We got higher accuracy. 
# %% [markdown]
# **Part b:** Part b. Baseline

# %%


# %% [markdown]
# ### 2) scikit-learn and tuning
# 
# Our goal will be to improve the tagger compared to the simple suffix-based tagger. For the further
# experiments, we move to scikit-learn which yields more options for considering various alternatives.
# We have reimplemented the ConsecutivePosTagger to use scikit-learn classifiers below. We have
# made the classifier a parameter so that it can easily be exchanged. We start with the BernoulliNBclassifier which should correspond to the way it is done in NLTK.

# %%
import numpy as np
import sklearn

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


class ScikitConsecutivePosTagger(nltk.TaggerI): 

    def __init__(self, train_sents, features=pos_features, clf = BernoulliNB()):
        # Using pos_features as default.
        self.features = features
        train_features = []
        train_labels = []
        for tagged_sent in train_sents:
            history = []
            untagged_sent = nltk.tag.untag(tagged_sent)
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = features(untagged_sent, i, history)
                train_features.append(featureset)
                train_labels.append(tag)
                history.append(tag)
        v = DictVectorizer()
        X_train = v.fit_transform(train_features)
        y_train = np.array(train_labels)
        clf.fit(X_train, y_train)
        self.classifier = clf
        self.dict = v

    def tag(self, sentence):
        test_features = []
        history = []
        for i, word in enumerate(sentence):
            featureset = self.features(sentence, i, history)
            test_features.append(featureset)
        X_test = self.dict.transform(test_features)
        tags = self.classifier.predict(X_test)
        return zip(sentence, tags)

# %% [markdown]
# **Part a)** Training the ScikitConsecutivePosTagger with *news_train* set and test on the *news_dev_test* set
# with the *pos_features*.

# %%
tagger_scikit = ScikitConsecutivePosTagger(news_train)
print(round(tagger_scikit.evaluate(news_dev_test), 4))

# %% [markdown]
# We can see that, by using the same data and same features we get a bit inferior results.
# %% [markdown]
# **Part b)** One explanation could be that the smoothing is too strong. *BernoulliNB()* from scikit-learn uses Laplace smoothing as default (“add-one”). The smoothing is generalized to Lidstone smoothing which is expressed by the alpha parameter to *BernoulliNB(alpha=…)*. Therefore, we will tune the alpha parameter to find the most optimal one. 

# %%
def tunning_bernoulli(pos_features):
    alphas = [1, 0.5, 0.1, 0.01, 0.001, 0.0001]
    accuracies = []
    for alpha in alphas:
        tagger_sci = ScikitConsecutivePosTagger(news_train,features = pos_features ,clf = BernoulliNB(alpha=alpha))
        accuracies.append(round(tagger_sci.evaluate(news_dev_test), 4))
    
    return alphas,accuracies
    
def visualize_results(alphas, accuracies):
    acc_alphas = {'alpha':alphas,'Accuracies':accuracies}
    import pandas as pd
    df = pd.DataFrame(acc_alphas)
    print(df)

    best_acc = max(accuracies)
    best_ind = accuracies.index(max(accuracies))
    best_alpha = alphas[best_ind]
    print("")
    print(f'Best alpha: {best_alpha} - accuracy: {best_acc}')


# %%
alphas,accuracies = tunning_bernoulli(pos_features)


# %%
visualize_results(alphas, accuracies)

# %% [markdown]
# We can see that we get a little bit better result with Scikits BernoulliNB with the best alpha.
# %% [markdown]
# **Part c)** To improve the results, we may change the feature selector or the machine learner. We start with
# a simple improvement of the feature selector. The NLTK selector considers the previous word, but
# not the word itself. Intuitively, the word itself should be a stronger feature. By extending the NLTK
# feature selector with a feature for the token to be tagged, we try to find the best results.

# %%
def pos_features_tagged(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}    
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        
    #same structure, but included the token to be tagged.
    features['tagged_word'] = sentence[i]

    return features


# %%
alphas_tag,accuracies_tag = tunning_bernoulli(pos_features_tagged)
visualize_results(alphas_tag, accuracies_tag)

# %% [markdown]
# ### 3) Logistic regression 
# %% [markdown]
# **Part a)** We proceed with the best feature selector from the last exercise. We will study the effect of the
# learner.

# %%
from sklearn.linear_model import LogisticRegression

#using solver='liblinear' since we are dealing with one-versus-rest schemes
logClf = LogisticRegression(solver= 'liblinear') 


# %%
tagger_log = ScikitConsecutivePosTagger(news_train,features = pos_features ,clf = logClf)
acc_log = (round(tagger_log.evaluate(news_dev_test), 4))
print(f'Logistic accuracy = {acc_log}')

# %% [markdown]
# The *Logistic Regression* classifier is better than all of the *BernoulliNB* methods without the token to be tagged.
# %% [markdown]
# **Part b)** Similarly to the Naive Bayes classifier, we will study the effect of smoothing. Smoothing for LogisticRegression is done by regularization. In scikit-learn, regularization is expressed by the parameter C. A smaller C means a heavier smoothing (C is the inverse of the parameter $\alpha$ in the lectures). We will tune the C parameter in order to find the most optimal model.

# %%
def tunning_logistic(pos_features):
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    accuracies = []
    for C in C_values:
        print(f"Running: LogisticRegression(C = {C})")
        logClf = LogisticRegression(C=C,solver= 'liblinear') 
        tagger_log = ScikitConsecutivePosTagger(news_train,features = pos_features ,clf = logClf)
        accuracies.append(round(tagger_log.evaluate(news_dev_test), 4))
    
    return C_values,accuracies


# %%
C_values,accuracies_log = tunning_logistic(pos_features)


# %%
visualize_results(C_values, accuracies_log)

# %% [markdown]
# We can see here that the best C value for the Logistic Regression is 0.01.
# %% [markdown]
# ### 4) Features
# %% [markdown]
# **Part a)** We will now stick to the LogisticRegression() with the optimal C from the last point and see
# whether we are able to improve the results further by extending the feature extractor with more
# features. First, try adding a feature for the next word in the sentence, and then train and test.

# %%
def pos_features_extended(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}    
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        
    #next word in the secquence:
    if i == len(sentence) - 1:
       features['next-word'] = sentence[i]
    else:
        features['next-word'] = sentence[i+1]
    
    return features


# %%
def find_accuracy(pos_features,news_train,news_dev_test):
    best_ind = accuracies_log.index(max(accuracies_log))
    optimal_C = C_values[best_ind]

    clf = LogisticRegression(C=optimal_C,solver= 'liblinear')
    tagger = ScikitConsecutivePosTagger(news_train,features = pos_features ,clf = clf)
    acc = (round(tagger.evaluate(news_dev_test), 4))
    return acc


# %%
acc_opt_log = find_accuracy(pos_features_extended,news_train,news_dev_test)
print(f'Logistic regression with optimal C: {acc_opt_log}')

# %% [markdown]
# **Part b)** We will continue to add more features to get an even better tagger. 

# %%
def pos_features_decapilized(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}    
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        
    #next word in the secquence:
    if i == len(sentence) - 1:
       features['next-word'] = sentence[i]
    else:
        features['next-word'] = sentence[i+1]
    features['current-word'] = sentence[i]  

    
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—'
    
    s = sentence[i]

    if s.isupper():
        s = s.lower()
    elif s.isdigit():
        features['type'] = 'digit'
    elif s in punctuation: 
        features['type'] = 'punctuation'
    else:
        features['type'] = 'other'

    return features


# %%
acc_extended = find_accuracy(pos_features_decapilized,news_train,news_dev_test)
print(f'Logistic regression with optimal C: {acc_extended}')

# %% [markdown]
# By adding the current word, we get very much more improvement.
# %% [markdown]
# ### 5) Larger corpus and evaluation
# 
# **Part a)** We will now test our best tagger so far on the news_test set.

# %%
acc_test_data = find_accuracy(pos_features_decapilized,news_train,news_test)
print(f'Logistic regression - accuracy = {acc_test_data}')

# %% [markdown]
# **Part b)** Now,we will use nearly the whole Brown corpus. But we will take away two categories for later evaluation: *adventure* and *hobbies*. We will also initially stay clear of *news* to be sure not to mix training and test data.

# %%
categories = brown.categories()
categories.remove('news')
categories.remove('adventure')
categories.remove('hobbies')
tagged_sents = brown.tagged_sents(categories=categories)
brown_data = brown.tagged_sents(categories = categories , tagset = 'unvisersal')

rest_test, rest_dev_test,rest_train = split_data(brown_data)


# %%
#merging the datasets

train = rest_train + news_train
test = rest_test + news_train
dev_test = rest_dev_test + news_dev_test

## establish baseline!!

# %% [markdown]
# **Part c)** We can then build our tagger for this larger domain. By using the best setting, we will try to find the accuracy for this dataset.

# %%
#acc_large = find_accuracy(pos_features_decapilized,train,test)

best_ind = accuracies_log.index(max(accuracies_log))
optimal_C = C_values[best_ind]

clf = LogisticRegression(C=optimal_C,solver= 'liblinear')
tagger_domain = ScikitConsecutivePosTagger(train,features = pos_features_decapilized ,clf = clf)
acc_domain = (round(tagger_domain.evaluate(test), 4))


# %%
print(f'The accuracy for the tagger for whole domain = {acc_domain}')


# %%
adventures = brown.tagged_sents(categories = 'adventures' , tagset = 'unvisersal')
hobbies = brown.tagged_sents(categories = 'hobbies' , tagset = 'unvisersal')


# %%
acc_adventure = (round(tagger_domain.evaluate(adventures), 4))
acc_hobbies = (round(tagger_domain.evaluate(hobbies), 4))


