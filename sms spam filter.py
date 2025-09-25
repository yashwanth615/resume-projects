#spam detection filter using nlp

'''
import nltk
#nltk.download_shell()
#importing the dataset (in built data)
#this part is just the analysis part
messages = [line.rstrip()for line in open('SMSSpamCollection')]
print(len(messages))
print(messages[1])
for mess_no,message in enumerate(messages[:10]):
	print(mess_no,message)
	print('\n')
import pandas as pd
messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
print(messages.head())
print(messages.describe())
#analyze the data from the description
print(messages.groupby('label').describe())
messages['length']=messages['message'].apply(len)
print(messages.head())
import matplotlib.pyplot as plt
import seaborn as sns 
messages['length'].plot.hist(bins=150)
plt.show()
print(messages['length'].describe())
print(messages[messages['length']==910]['message'].iloc[0])
messages.hist(column='length',by='label',bins=100,figsize=(12,6))
plt.show()
#text pre processing converting words into vectors
import string
#here is how we actually take and process the text data
#lets start by removing the punctuation
ex_message='sample message! notice: it has punctuation.'
ex_message_nopunc=[c for c in ex_message if c not in string.punctuation]
print(ex_message_nopunc)
#joining the letters back
ex_message_nopunc=''.join(ex_message_nopunc)
print(ex_message_nopunc)
#demonstration for join method
'''
'''
x=['a','b','c','d']
print(''.join(x))
'''
'''
#now we are removing unwanted words from the messages retaining only the key words
from nltk.corpus import stopwords
#stop words is the set of commonly used words which are not key words in the message and hence we should get rid of such words 
ex_message_nopunc=ex_message_nopunc.split()
print(ex_message_nopunc)
clean_message=[word for word in ex_message_nopunc if word.lower() not in stopwords.words('english')]
print(clean_message)
#now we have to perform all these operations on all the messages 
def text_process(mess):
     no_punc=[char for char in mess if char not in string.punctuation]
     no_punc=''.join(no_punc)
     return[word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
print(messages['message'].head(5).apply(text_process))     
#now creating the sparse matrix(bag of words)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow_transformer.get_feature_names_out()[4068])
messages_bow=bow_transformer.transform(messages['message'])
print('shape of the sparse matrix:',messages_bow.shape)
print(messages_bow.nnz)
sparsity=(100.0*messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
print('sparsity:{}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
# if i had to check the term frequency or the inverse document frequency the it can be done as follows
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
messages_tfidf = tfidf_transformer.transform(messages_bow)
# now we are going to use naive_bayes classification theorem for classifying our theorem
from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(messages_tfidf,messages['label'])
#testing it with a single message 
print(spam_detection_model.predict(tfidf4)[0])
print(messages['label'][3])
all_predictions=spam_detection_model.predict(messages_tfidf)
print(all_predictions)
#splitting our data to evaluate our model
from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test=train_test_split(messages['message'],messages['label'],test_size=0.3)
#after splitting the data we might have to undergo all these steps again 
'''
'''
but its not necessary we can use the pipeline feature ability to perform this task
'''
'''
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('bow_transformer',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),('classifier',MultinomialNB())])
pipeline.fit(msg_train, label_train)
predictions =pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))

'''

'''
nlp project
'''

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('yelp.csv')
print(df.head())
print(df.info())
print(df.describe())
df['text_length']=df['text'].apply(len)
print(df.head())
g=sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text_length',bins=50)
plt.show()
sns.barplot(x='stars',y='text_length',data=df)
plt.show()
sns.countplot(x='stars',data=df)
plt.show()
stars= df.groupby('stars').mean()
print(stars)
stars_corr=stars.corr()
print(stars_corr)
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()
