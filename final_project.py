#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import lbsa
import os 
import math
import torch

import warnings
warnings.simplefilter("ignore", UserWarning)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = get_default_device()
device


# In[2]:


train_df = pd.read_csv("Corona_NLP_train.csv", encoding='latin-1')
test_df = pd.read_csv("Corona_NLP_test.csv", encoding='latin-1')


# In[3]:


test_df.shape


# In[4]:


os.getcwd().replace("\\", "/")


# In[5]:


sa_lexicon = lbsa.get_lexicon("sa", language = "english", source = "nrc", 
                              path = "E:/Aalto/Aalto Master 1st Spring 2022/DSB2/Assignment 4/Lexicon-Based-Sentiment-Analysis/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx")


# In[6]:


sa_lexicon.dataframe


# In[7]:


sa_lexicon.dataframe[["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]].sum(axis=0)


# In[8]:


train_df.head()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="darkgrid")
sns.set(font_scale=1.3)

target_dist = sns.catplot(x='Sentiment', data=train_df, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
plt.show()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="darkgrid")
sns.set(font_scale=1.3)

target_dist = sns.catplot(x='Sentiment', data=test_df, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
plt.show()


# In[11]:


palette=sns.color_palette('magma')
sns.set(palette=palette)
plt.figure(figsize=(12,6))
plt.title('Top 10 cities with highest Tweets')
countries =sns.countplot(x='Location', data=train_df, order=train_df['Location'].value_counts().index[:10], palette=palette)
countries.set_xticklabels(countries.get_xticklabels(), rotation=45)
plt.show() 


# In[12]:


train_df.head()


# In[13]:


'''
from wordcloud import WordCloud
for label, cmap in zip(['Positive', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative'],
                       ['winter', 'autumn', 'magma', 'viridis', 'plasma']):
    text = train_df.query('Sentiment == @label')['OriginalTweet'].str.cat(sep=' ')
    plt.figure(figsize=(10, 6))
    wc = WordCloud(width=1000, height=600, background_color="#f8f8f8", colormap=cmap)
    wc.generate_from_text(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.title(f"Words Commonly Used in ${label}$ Messages", size=20)
    plt.show()
    '''


# ## a) Extract lexicon-based features. Describe whether there is any lexicon-based feature that has some level of relation to the sentiment. Explain your reasoning.

# In[14]:


afinn_lexicon = lbsa.get_lexicon('opinion', language='english', source='afinn')
nrc_lexicon = lbsa.get_lexicon('opinion', language='english', source='nrc')
nrc_sa_lexicon = lbsa.get_lexicon('sa', language='english', source='nrc')
mpqa_lexicon = lbsa.get_lexicon('opinion', language='english', source='mpqa')

sa_extractor = lbsa.FeatureExtractor(afinn_lexicon, nrc_lexicon, nrc_sa_lexicon, mpqa_lexicon)
feat_dat_train = pd.DataFrame(sa_extractor.process(train_df['OriginalTweet']),columns=sa_extractor.feature_names)
feat_dat_test = pd.DataFrame(sa_extractor.process(test_df['OriginalTweet']),columns=sa_extractor.feature_names)


# ## Apply Data Preprocessing onto also the test_df 

# In[15]:


train_df["Sentiment"] = train_df["Sentiment"].map({'Extremely Negative':"Negative",'Negative':"Negative",'Neutral':"Neutral",'Positive':"Positive",'Extremely Positive': "Positive"})
test_df["Sentiment"] = test_df["Sentiment"].map({'Extremely Negative':"Negative",'Negative':"Negative",'Neutral':"Neutral",'Positive':"Positive",'Extremely Positive': "Positive"})


# In[16]:


def show_dist(df, col):
    print('Descriptive stats for {}'.format(col))
    print('-'*(len(col)+22))
    print(df.groupby('Sentiment')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='Sentiment', height=5, hue='Sentiment', palette="PuBuGn_d")
    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)
    plt.show()

df_eda_train = pd.concat([train_df,feat_dat_train],axis=1)
df_eda_test = pd.concat([test_df,feat_dat_test],axis=1)

'''
for feature in sa_extractor.feature_names:
    show_dist(df_eda, feature)
'''


# ## b) Clean the text. Describe each step that you went through and explain why those steps are needed. (Perform at least 5 steps)

# ## Step 3: Cleaning the text - a traditional approach
# 
# Text cleaning often includes following steps
# - lowercasing
# - removal of punctuations, question and exclamation marks
# - removal of urls
# - removal of digits
# - removal of stopwords
# - apply stemming (e.g., PorterStemmer) or lemmatization
# (https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)

# In[17]:


from sklearn.base import BaseEstimator, TransformerMixin

import re, string, os, emoji

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class CleanText(BaseEstimator, TransformerMixin):
   
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub(r'\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X


# In[18]:


ct = CleanText()
sent_clean = ct.fit_transform(train_df.OriginalTweet)
print(sent_clean.sample(5))
empty_clean = sent_clean == ''
print('{} records have no words left after text cleaning'.format(sent_clean[empty_clean].count()))
sent_clean.loc[empty_clean] = '[no_text]'


# In[19]:


sent_clean_test = ct.transform(test_df.OriginalTweet)
empty_clean_test = sent_clean_test == ''
print('{} records have no words left after text cleaning'.format(sent_clean_test[empty_clean_test].count()))
sent_clean_test.loc[empty_clean_test] = '[no_text]'


# ## c) Create BOW representation of the data. Identify the most common words and visualize. Discuss whether the result is reasonable.

# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
import collections

cv = CountVectorizer()
bow = cv.fit_transform(sent_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(20, 15))
bar_freq_word = sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show();


# In[21]:


#sent_clean


# In[22]:


df_model = df_eda_train
df_model['clean_text'] = sent_clean
#df_model.columns.tolist()


# In[23]:


test_df["clean_text"] = sent_clean_test


# In[24]:


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df_model.drop('Sentiment', axis=1), df_model.Sentiment, test_size=0.1, random_state=30)


# In[26]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

mnb = MultinomialNB()
logreg = LogisticRegression(max_iter=10000, solver='saga')
countvect = CountVectorizer()


# ## d) Discuss at least two other types of representations than the BOW(or 1-hot encoding) in terms of their principles and the areas of applications. Bonus point: Implement one of them with python code and present the result. (4 points + 4 bonus points)

# 2 other types of representations: 
# - TF-IDF
# - Word embedding: 

# In[27]:


print("First sentence:", df_model.head(3).OriginalTweet[1])
print()
print("Second sentence:", df_model.head(3).OriginalTweet[2])


# In[28]:


corpus = df_model.head(3).clean_text.tolist()
corpus


# In[29]:


first = df_model.head(3).clean_text.tolist()[1]
second = df_model.head(3).clean_text.tolist()[2]

#split so each word have their own string
first = first.split(" ")
second= second.split(" ")

#join them to remove common duplicate words
total= set(first).union(set(second))

print(total)

#Now lets add a way to count the words using a dictionary key-value pairing for both sentences
wordDictA = dict.fromkeys(total, 0) 
wordDictB = dict.fromkeys(total, 0)
for word in first:
    wordDictA[word]+=1
    
for word in second:
    wordDictB[word]+=1
#put them in a dataframe and then view the result:
pd.DataFrame([wordDictA, wordDictB])


# In[30]:


#Now writing the TF function:
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict
#running our sentences through the tf function:
tfFirst = computeTF(wordDictA, first)
tfSecond = computeTF(wordDictB, second)
#Converting to dataframe for visualization
tf_df= pd.DataFrame([tfFirst, tfSecond])

tf_df


# In[32]:


#creating the log portion of the Excel table we saw earlier
def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict
#inputing our sentences in the log file
idfs = computeIDF([wordDictA, wordDictB])
#The actual calculation of TF*IDF from the table above:
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf
#running our two sentences through the IDF:
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
#putting it in a dataframe
idf= pd.DataFrame([idfFirst, idfSecond])


# ## e) Train classifiers. Describe the process in two sections, one for the methods and the other for the results.

# In[33]:


from pprint import pprint
from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def grid_vect(clf, parameters_clf, X_train, X_test, y_train, y_test, parameters_text=None, vect=None):
    
    textcountscols = ['afinn_positive', 'afinn_negative', 'nrc_positive', 'nrc_negative', 'nrc_anger', 'nrc_anticipation', 'nrc_disgust',
                     'nrc_fear', 'nrc_joy', 'nrc_sadness', 'nrc_surprise', 'nrc_trust', 'mpqa_positive', 'mpqa_negative', 'mpqa_strong_subjectivty']
    
    features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                 , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                                , n_jobs=-1)
  
    pipeline = Pipeline([
        ('features', features)
        , ('clf', clf)
    ])
    
    # Join the parameters dictionaries together
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)

    # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
                        
    return grid_search


# In[34]:


# Parameter grid settings for the vectorizers (Count and TFIDF)
parameters_vect = {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
    'features__pipe__vect__min_df': (1,2)
}

# Parameter grid settings for LogisticRegression
parameters_logreg = {
    'clf__C': (0.25, 0.5, 1.0),
    'clf__penalty': ('l1', 'l2')
}

# Parameter grid settings for MultinomialNB
parameters_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
}


# In[35]:


y_train.value_counts()


# In[36]:


y_valid.value_counts()


# In[37]:


# LogisticRegression
best_logreg_countvect = grid_vect(logreg, parameters_logreg, X_train, X_valid, y_train, y_valid, parameters_text=parameters_vect, vect=countvect)


# In[39]:


mnb = MultinomialNB()
countvect = CountVectorizer()

# MultinomialNB
best_mnb_countvect = grid_vect(mnb, parameters_mnb, X_train, X_test, y_train, y_test, parameters_text=parameters_vect, vect=countvect)


# ## BERT

# In[41]:


#general purpose packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#data processing
import re, string
import emoji
import nltk

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

#keras
import tensorflow as tf
from tensorflow import keras


#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

#set seed for reproducibility
seed=42

#set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)


# ### Import the BERT tokenizer

# In[43]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# '''
# token_lens = []
# 
# for txt in df['text_clean'].values:
#     tokens = tokenizer.encode(txt, max_length=512, truncation=True)
#     token_lens.append(len(tokens))
#     
# max_len=np.max(token_lens)
# '''

# ## Class Balancing by RandomOverSampler

# In[46]:


ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(np.array(X_train['clean_text']).reshape(-1, 1), np.array(y_train).reshape(-1, 1));
#train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text_clean', 'Sentiment']);


# In[47]:


X_valid, y_valid = ros.fit_resample(np.array(X_valid['clean_text']).reshape(-1, 1), np.array(y_valid).reshape(-1, 1));


# ## Train- Validation split

# In[52]:


y_test = test_df.Sentiment


# In[53]:


ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()


# In[54]:


print(f"TRAINING DATA: {X_train.shape[0]}\nVALIDATION DATA: {X_valid.shape[0]}\nTESTING DATA: {test_df.shape[0]}" )


# ## BERT Sentiment Analysis

# In[55]:


MAX_LEN=128


# In[56]:


def tokenize(data,max_len=MAX_LEN) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


# In[57]:


train_input_ids, train_attention_masks = tokenize(pd.DataFrame(X_train,columns=["clean_text"]).clean_text.tolist(), MAX_LEN)
val_input_ids, val_attention_masks = tokenize(pd.DataFrame(X_valid,columns=["clean_text"]).clean_text.tolist(), MAX_LEN)
test_input_ids, test_attention_masks = tokenize(pd.DataFrame(test_df, columns=["clean_text"]).clean_text.tolist(), MAX_LEN)


# In[59]:


bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# In[60]:


def create_model(bert_model, max_len=MAX_LEN):
    
    ##params###
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()


    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    embeddings = bert_model([input_ids,attention_masks])[1]
    
    output = tf.keras.layers.Dense(3, activation="softmax")(embeddings)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks], outputs = output)
    
    model.compile(opt, loss=loss, metrics=accuracy)
    
    
    return model


# ## With balanced data

# In[ ]:


model_bal = create_model(bert_model, MAX_LEN)
model_bal.summary()


# In[62]:


history_bert = model_bal.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_valid), epochs=5, batch_size=16)


# In[64]:


result_bert = model_bal.predict([test_input_ids,test_attention_masks])


# In[65]:


y_pred_bert =  np.zeros_like(result_bert)
y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1


# In[66]:


def conf_matrix(y, y_pred, title):
    fig, ax =plt.subplots(figsize=(5,5))
    labels=['Negative', 'Neutral', 'Positive']
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size":25})
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17) 
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.show()


# In[67]:


conf_matrix(y_test.argmax(1), y_pred_bert.argmax(1),'BERT Sentiment Analysis\nConfusion Matrix')


# In[68]:


print('\tClassification Report for BERT:\n\n',classification_report(y_test,y_pred_bert, target_names=['Negative', 'Neutral', 'Positive']))


# ## With imbalanced data

# In[54]:


model = create_model(bert_model, MAX_LEN)
model.summary()


# In[60]:


history_bert = model.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_valid), epochs=5, batch_size=16)


# In[61]:


result_bert = model.predict([test_input_ids,test_attention_masks])


# In[62]:


y_pred_bert =  np.zeros_like(result_bert)
y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1


# In[64]:


conf_matrix(y_test.argmax(1), y_pred_bert.argmax(1),'BERT Sentiment Analysis\nConfusion Matrix')


# In[65]:


print('\tClassification Report for BERT:\n\n',classification_report(y_test,y_pred_bert, target_names=['Negative', 'Neutral', 'Positive']))

