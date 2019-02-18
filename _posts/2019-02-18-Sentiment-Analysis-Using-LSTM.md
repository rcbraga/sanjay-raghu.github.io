---
title: "Sentiment Analysis using LSTM model, Class imbalance problem, Keras with Scikit Learn"
tags:
  - sentiment analysis
  - text analysis
  - keras
  - scikit learn
  - data imbalance
  - weighted loss fuction
---
[Github](https://github.com/sanjay-raghu) <br/>
[Linkedin](https://www.linkedin.com/in/sanjayiitg/) <br/>
**Sentiment Analysis:** the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.

**What's New** I have added how to deal with data imbalance. Almost all classification task have this problem as number of data of every class if different. For current dataset number of data having positive sentiments is very low relative to data with negative sentiment.

**Solving class imbalaned data**:
- upsampling 
- using class weighted loss function

## Dataset
First GOP Debate Twitter Sentiment
About this Dataset
This data originally came from [Crowdflower's Data for Everyone library](http://www.crowdflower.com/data-for-everyone).

As the original source says,
We looked through tens of thousands of tweets about the early August GOP debate in Ohio and asked contributors to do both
sentiment analysis and data categorization. Contributors were asked if the tweet was relevant, which candidate was mentioned,
what subject was mentioned, and then what the sentiment was for a given tweet. We've removed the non-relevant messages from
the uploaded dataset.

## **Details about model**

 - model contains 3 layers (Embedding, LSTM, Dense with softmax).
 - Upsampling is used to balance the data of minority class.
 - Loss fuction with different class weight in keras to further reduce class imbalance.

## Lets start coding
### Importing usefull packages
Lets first import all import libraries that will be used. 

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import re

```
### Data Preprocessing
- reading the data
- kepping only neccessary columns
- droping "Neutral" sentiment data

```python
path = "./Sentimen.csv" # change to the path to the Sentiment.csv file
data = pd.read_csv(path)
data = data[['text','sentiment']]
data = data[data.sentiment != "Neutral"]
```

Let See the few lines of the data
```python
data.head()
```

output will be something like this

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RT @NancyLeeGrahn: How did everyone feel about...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RT @ScottWalker: Didn't catch the full #GOPdeb...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RT @TJMShow: No mention of Tamir Rice and the ...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RT @RobGeorge: That Carly Fiorina is trending ...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RT @DanScavino: #GOPDebate w/ @realDonaldTrump...</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>

A few things to notice here
- "RT @..." in start of every tweet
- a lot of special characters <br>
We have to remove all this noise also lets convert text into lower case.

```python
data['text'] = data['text'].apply(lambda x: x.lower()) #lower caseing
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) # removing special chars
data['text'] = data.text.str.replace('rt','') # removing "rt"
```

Lets see the data again
```python
data.head()
```

You should see something like this

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>scottwalker didnt catch the full gopdebate la...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>robgeorge that carly fiorina is trending  hou...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>danscavino gopdebate w realdonaldtrump delive...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gregabbott_tx tedcruz on my first day i will ...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>warriorwoman91 i liked her and was happy when...</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>

This looks better.<br>
Lets pre-process the data so that we can use it to train the model
- Tokenize
- Padding (to make all sequence of same lengths)
- Converting sentiments into numerical data(One-hot form)
- train test split


```python
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
# print(X[:2])

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)

```
Next, I compose the LSTM Network. Note that **embed_dim**, **lstm_out**, **batch_size**, **droupout_x** variables are hyper parameters, their values are somehow intuitive, can be and must be played with in order to achieve good results. Please also note that I am using softmax as activation function. The reason is that our Network is using categorical crossentropy, and softmax is just the right activation method for that.

```python
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 28, 128)           256000    
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 28, 128)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 196)               254800    
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 394       
=================================================================
Total params: 511,194
Trainable params: 511,194
Non-trainable params: 0
_________________________________________________________________
None


