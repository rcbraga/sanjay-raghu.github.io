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

Layer (type)         |        Output Shape        |      Param #   
----------------------|---------------------------|----------------
embedding_1 (Embedding)   |   (None, 28, 128)     |      256000    
spatial_dropout1d_1 (Spatial) |(None, 28, 128)     |      0         
lstm_1 (LSTM)       |         (None, 196)          |     254800    
dense_1 (Dense)      |        (None, 2)             |    394       
Total params: 511,194 | |
Trainable params: 511,194 | |
Non-trainable params: 0 | |


Here we train the Network. We should run much more than 15 epoch, but I would have to wait forever (run it later), so it is 15 for now.
```python
batch_size = 128
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1)
```
you will see progress bar (if you want to shut it up use verbose = 0)

Lets test the model with test data

```python
Y_pred = model.predict_classes(X_test,batch_size = batch_size)
df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
# print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))
report = classification_report(df_test.true, df_test.pred)
print(report)
```

class  |  precision  |  recall | f1-score  | support
-------|-------------|---------|-----------|--------------
   0   |    0.88     |   0.91  |    0.90   |  1713
   1   |    0.60     |   0.51  |    0.55   |  433

avg / total    |   0.82  |    0.83    |  0.83    |  2146

It is clear that finding negative tweets (**class 0**) goes very well (**recall 0.92**) for the Network but deciding whether is positive (**class 1**) is not really (**recall 0.52**). My educated guess here is that the positive training set is dramatically smaller than the negative, hence the "bad" results for positive tweets.

## Solving data imbalance problem

**1. Up-sample Minority Class**

Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.
There are several heuristics for doing so, but the most common way is to simply resample with replacement.

It's important that we separate test set before upsampling because after upsampling there will be multiple copies of
same data point and if we do train test split after upsamling the test set will not be compleatly unseen.  

```python
# Separate majority and minority classes

data_majority = data[data['sentiment'] == 'Negative']
data_minority = data[data['sentiment'] == 'Positive']

bias = data_minority.shape[0]/data_majority.shape[0]
# lets split train/test data first then 
train = pd.concat([data_majority.sample(frac=0.8,random_state=200),
         data_minority.sample(frac=0.8,random_state=200)])
test = pd.concat([data_majority.drop(data_majority.sample(frac=0.8,random_state=200).index),
        data_minority.drop(data_minority.sample(frac=0.8,random_state=200).index)])

train = shuffle(train)
test = shuffle(test)

print('positive data in training:',(train.sentiment == 'Positive').sum())
print('negative data in training:',(train.sentiment == 'Negative').sum())
print('positive data in test:',(test.sentiment == 'Positive').sum())
print('negative data in test:',(test.sentiment == 'Negative').sum())

```
positive data in training: 1789 <br>
negative data in training: 6794 <br>
positive data in test: 447 <br>
negative data in test: 1699 <br>

Now Lets do up-sampling

```python
# Separate majority and minority classes in training data for upsampling 
data_majority = train[train['sentiment'] == 'Negative']
data_minority = train[train['sentiment'] == 'Positive']

print("majority class before upsample:",data_majority.shape)
print("minority class before upsample:",data_minority.shape)

# Upsample minority class
data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
 
# Display new class counts
print("After upsampling\n",data_upsampled.sentiment.value_counts(),sep = "")
```
Output:
```
majority class before upsample: (6794, 2)
minority class before upsample: (1789, 2)
After upsampling
Positive    6794
Negative    6794

```