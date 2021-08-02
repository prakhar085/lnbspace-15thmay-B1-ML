#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from warnings import filterwarnings as fw
fw('ignore')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer


# In[2]:


data = pd.read_csv('SMSSpamCollection.tsv', sep = '\t', names = ['type', 'msg'])
data


# In[3]:


data.info()


# In[4]:


data.describe().T


# In[5]:


data.groupby('type').describe()


# In[6]:


for i in data['msg'][data['type']=='ham']:
    print(i)
    print('-')


# In[7]:


data['length'] = data['msg'].str.len()
data.sample(5)


# In[8]:


data.groupby('type').describe()


# In[9]:


data['word_count'] = data['msg'].str.split().str.len()
data.sample(5)


# In[10]:


data.groupby('type').describe().T


# In[11]:


data['msg'][(data['word_count'] == 2) & (data['type'] == 'spam')]


# In[12]:


mm = data['msg'][data['length'] == max(data['length'])].values[0]
mm


# In[13]:


data['msg'][data['length'] > 100]


# In[14]:


m7 = data['msg'][7]
m7


# In[15]:


lemma = WordNetLemmatizer()
stemmer = PorterStemmer()


# In[16]:


sw = ["i've", "i'll", "we'll", "ve"]


# In[47]:


def text_process(t):
    all_words = []
    t = t.replace('.', '. ').replace('!', '! ').replace('?', '? ')
    for sent in nltk.sent_tokenize(t.lower()):
        words = nltk.word_tokenize(sent)
        words = [word for word in words if (word not in stopwords.words('english')) and (word not in sw)]
#         print(words)
#         words = [lemma.lemmatize(word, wordnet.ADJ) for word in words]
        words = [stemmer.stem(word) for word in words]
        words = [word for word in words if word not in punctuation]
#         print(words)
        words = [word for word in words if not word.isnumeric()]
        all_words += words
#     print(all_words)
    all_words = ''.join([ch for ch in ' '.join(all_words) if (ch not in punctuation) and (not ch.isnumeric())])
    return all_words


# In[18]:


m7


# In[19]:


text7 = text_process(m7)
print(text7)


# In[20]:


print(mm)
print()
textm = text_process(mm)
print(textm)
# . ke baad space nahi h to i alag nahi horha 
# to . ke baad space laga det hain


# In[21]:


data['msg'].head(4).apply(text_process)


# In[22]:


"i've" in stopwords.words('english')


# In[23]:


# i've stopwords me nahi h haww


# In[48]:


tfidfV = TfidfVectorizer()
tf_idf = tfidfV.fit_transform(data['msg'].apply(text_process))


# In[25]:


tf_df = pd.DataFrame(tf_idf.toarray(), columns = tfidfV.get_feature_names())
tf_df.head()


# In[26]:


tf_df.columns[0].isnumeric()


# In[27]:


tfidfV.get_feature_names()


# In[28]:


tf_df.shape


# In[29]:


# whenever dealing with text data use naive bayes


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


xtrain, xtest, ytrain, ytest = train_test_split(tf_df, data['type'], test_size = 0.25, random_state = 0)


# In[32]:


from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report


# In[33]:


model_bnb = BernoulliNB().fit(xtrain, ytrain)
print(model_bnb.score(xtrain, ytrain))
print(model_bnb.score(xtest, ytest))


# In[34]:


model_bnb = MultinomialNB().fit(xtrain, ytrain)
print(model_bnb.score(xtrain, ytrain))
print(model_bnb.score(xtest, ytest))


# In[35]:


# next project
# web scraping -> some mobiles reviews 
# not .text -> we have find out attributes to go on a particular link
# anchor tag ka attribute h href
# instead of .text we have to use attributes


# In[36]:


# collect reviews and stars
# try to build a system to predict ki usko product kesa laga
# sirf review daalta h to hamko predict krna h ki neg reviw tha ya pos rating tha
# atleast 10 pages ka


# In[37]:


xtrainy, xtesty, ytrainy, ytesty = train_test_split(data['msg'], data['type'], test_size = 0.25, random_state = 0)


# In[38]:


from sklearn.pipeline import Pipeline


# In[39]:


ddd =map(text_process, data['msg'])


# In[40]:


TfidfVectorizer().fit_transform(ddd)


# In[41]:


model_pipe = Pipeline([
    ('bow', CountVectorizer(analyzer = text_process)),
    ('tfidf', TfidfTransformer()),
    ('clf', BernoulliNB())
])


# In[42]:


model_pipe.fit(xtrainy, ytrainy)


# In[43]:


model_pipe.score(xtrainy, ytrainy)


# In[44]:


model_pipe.score(xtesty, ytesty)


# In[45]:


ypredy = model_pipe.predict(xtesty)


# In[46]:


print(classification_report(ytesty, ypredy))


# In[ ]:




