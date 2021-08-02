#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings as fw
fw('ignore')


# In[2]:


df = pd.read_csv('news.csv')
df


# In[3]:


del df['Unnamed: 0']


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df['title-length'] = df['title'].str.len()
df['text-length'] = df['text'].str.len()


# In[7]:


df


# In[8]:


df.groupby('label').describe().T


# In[9]:


sns.set_style('darkgrid')
sns.countplot(x = 'label', data = df)


# ## Data Cleansing and Preprocessing
# 1. Removing ,!? etc using Regex
# 2. Converting to lower case
# 3. Spliting the messages into words
# 4. Removing the stop words using stopwords and punctuations(if any left)
# 5. Applying lematization/stemming to words (excluding stop words)
# 6. Joining the words back into a sentence

# In[10]:


import nltk
from string import punctuation
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
lemma = WordNetLemmatizer()
stemmer = PorterStemmer()


# In[11]:


def clean_text(t):
    t = re.sub(r'[^a-zA-Z]', ' ', t)
    t = t.lower()
    t = ' '.join([stemmer.stem(word) for word in t.split() if (word not in stopwords.words('english')) and (word not in punctuation)])
    return t


# In[12]:


df['Clean_title'] = df['title'].apply(clean_text)
df['Clean_text'] = df['text'].apply(clean_text)


# In[13]:


df.head()


# ## Visualing Data

# In[14]:


from wordcloud import WordCloud


# In[21]:


fake_words = ' '.join(list(df['text'][df['label'] == 'FAKE']))
real_words = ' '.join(list(df['text'][df['label'] == 'REAL']))


# In[22]:


real_wc = WordCloud(background_color="white", max_words=len(real_words)).generate(real_words)
plt.figure(figsize =(15, 10))
plt.imshow(real_wc)
plt.axis('off')
plt.suptitle('Real Words', fontweight = 'bold', fontsize = 20)


# In[23]:


fake_wc = WordCloud(background_color="white", max_words=len(fake_words), colormap='gist_heat').generate(fake_words)
plt.figure(figsize =(15, 10))
plt.imshow(fake_wc)
plt.axis('off')
plt.suptitle('Fake Words', fontweight = 'bold', fontsize = 20)


# ### Transforming text to feature vectors that can be used as input to estimator

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[25]:


tfidfV_title = TfidfVectorizer()
tfidfV_text = TfidfVectorizer()


# In[41]:


# tf_idf_title = tfidfV_title.fit_transform(df['Clean_title'])
tf_idf_text = tfidfV_text.fit_transform(df['Clean_text'])


# In[42]:


# tf_df_title = pd.DataFrame(tf_idf_title.toarray(), columns = tfidfV_title.get_feature_names())
tf_df_text = pd.DataFrame(tf_idf_text.toarray(), columns = tfidfV_text.get_feature_names())


# In[28]:


tf_df_title.shape


# In[29]:


tf_df_text.shape


# In[30]:


tf_df = pd.concat([tf_df_title, tf_df_text], axis = 1)


# In[31]:


tf_df


# In[43]:


# tf_df = tf_df.astype('uint8')
tf_df_text = tf_df_text.astype('uint8')


# ## Model Building

# In[32]:


from sklearn.model_selection import train_test_split


# In[44]:


x = tf_df_text
y = df['label']


# In[34]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import scikitplot as skplt


# In[35]:


def models(model, x, y):
#     splitting the data
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)
    
    model.fit(xtrain, ytrain)
    
    ypred = model.predict(xtest)
    y_probas = model.predict_proba(xtest)
    
    cv_score = cross_val_score(model,x,y,cv= 10)
    
    print('Training Score: ', round((model.score(xtrain, ytrain))*100, 2), '%')
    print('Testing Score: ', round((model.score(xtest, ytest))*100, 2), '%')
    print("Cross Val scoe: ", round((np.mean(cv_score)*100), 2), '%')
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))
    skplt.metrics.plot_roc(ytest, y_probas, figsize=(12,8), title_fontsize=12, text_fontsize=16)
    plt.show()
    skplt.metrics.plot_precision_recall(ytest, y_probas, figsize=(12,8), title_fontsize=12, text_fontsize=16)
    plt.show()


# In[45]:


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
models(model, x, y)


# In[ ]:




