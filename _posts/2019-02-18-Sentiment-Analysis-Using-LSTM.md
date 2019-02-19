---
title: "Sentiment Analysis using LSTM model, Class Imbalance Problem, Keras with Scikit Learn"
tags:
  - sentiment analysis
  - text analysis
  - keras
  - scikit learn
  - data imbalance
  - weighted loss fuction
---
The code in this post can be found at my [Github](https://github.com/sanjay-raghu) repository. If you are also interested in trying out the code
I have also written a code in Jupyter Notebook form on [Kaggle](https://www.kaggle.com/sanjay11100/lstm-sentiment-analysis-data-imbalance-keras) there you
don't have to worry about installing anything just run Notebook directly.   



**Sentiment Analysis:**<br> 
The process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral. In common ML words its just a classification problem. 

**What is class imbalance:**<br>
It is the problem in machine learning where the total number of a class of data (positive) is far less than the total number of another class of data (negative). This problem is extremely common in practice and can be observed in various disciplines including fraud detection, anomaly detection, medical diagnosis, oil spillage detection, facial recognition, etc.

If you want to know more about class imbalance problem, [here](http://www.chioka.in/class-imbalance-problem/) is a link of a great blog post

**Solving class imbalanced data:**<br>
I am using the two most effective ways to mitigate this:<br>
- Up sampling 
- Using class weighted loss function

**Dataset**<br>
First GOP Debate Twitter Sentiment
About this Dataset
This data originally came from [Crowdflower's Data for Everyone library ](http://www.crowdflower.com/data-for-everyone).

> As the original source says,
> We looked through tens of thousands of tweets about the early August GOP debate in Ohio and asked contributors to do both
> sentiment analysis and data categorization. Contributors were asked if the tweet was relevant, which candidate was mentioned,
> what subject was mentioned, and then what the sentiment was for a given tweet. We've removed the non-relevant messages from
> the uploaded dataset.

**Details about model**<br>
 - model contains 3 layers (Embedding, LSTM, Dense with softmax).
 - Up-sampling is used to balance the data of minority class.
 - Loss function with different class weight in keras to further reduce class imbalance.
## Lets start coding
### Importing useful packages
Lets first import all libraries. Please make sure that you have these libraries installed.   

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

---

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

> A few things to notice here
- "RT @..." in start of every tweet
- a lot of special characters <br>
> We have to remove all this noise also lets convert text into lower case.

```python
data['text'] = data['text'].apply(lambda x: x.lower()) #lower caseing
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) # removing special chars
data['text'] = data.text.str.replace('rt','') # removing "rt"
```

Lets see the data again
```python
data.head()
```

> You should see something like this

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

---
### Defining model
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

---
### Let's train the model

Here we train the Network. We should run much more than 15 epoch, but I would have to wait forever (run it later), so it is 15 for now.
```python
batch_size = 128
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1)
```
you will see progress bar (if you want to shut it up use verbose = 0)

Lets test the model with test data


### Let evaluate the model

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

> It is clear that finding negative tweets (**class 0**) goes very well (**recall 0.92**) for the Network but deciding whether is positive (**class 1**) is not really (**recall 0.52**). My educated guess here is that the positive training set is dramatically smaller than the negative, hence the "bad" results for positive tweets.

-----
## Solving data imbalance problem

**1. Up-sample Minority Class**

Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal. There are several heuristics for doing so, but the most common way is to simply re-sample with replacement.

It's important that we separate test set before up-sampling because after up-sampling there will be multiple copies of same data point and if we do train test split after up-sampling the test set will not be completely unseen.

```python
# Separate majority and minority classes
data_majority = data[data['sentiment'] == 'Negative']
data_minority = data[data['sentiment'] == 'Positive']

# will be used later in defining class weights
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
# Separate majority and minority classes in training data for up sampling 
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
Let's repeat the preprocessing step and define model again

```python
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values) # training with whole data

X_train = tokenizer.texts_to_sequences(data_upsampled['text'].values)
X_train = pad_sequences(X_train,maxlen=29)
Y_train = pd.get_dummies(data_upsampled['sentiment']).values
print('x_train shape:',X_train.shape)

X_test = tokenizer.texts_to_sequences(test['text'].values)
X_test = pad_sequences(X_test,maxlen=29)
Y_test = pd.get_dummies(test['sentiment']).values
print("x_test shape", X_test.shape)

# model
embed_dim = 128
lstm_out = 192

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
```

Lets define class weights as a dictionary, I have defined weight of majority class to be 1 and
of minority class to be a multiple of $\frac{1}{bias}$

```python
batch_size = 128
# also adding weights
class_weights = {0: 1 ,
                1: 1.6/bias }
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1,
          class_weight=class_weights)

```

Epoch 1/15
13588/13588 [==============================] - 10s 700us/step - loss: 1.2697 - acc: 0.5703
Epoch 2/15
13588/13588 [==============================] - 9s 627us/step - loss: 0.7914 - acc: 0.7612
Epoch 3/15
13588/13588 [==============================] - 8s 621us/step - loss: 0.6597 - acc: 0.8159
Epoch 4/15
13588/13588 [==============================] - 8s 623us/step - loss: 0.5813 - acc: 0.8403
Epoch 5/15
13588/13588 [==============================] - 8s 621us/step - loss: 0.5450 - acc: 0.8534
Epoch 6/15
13588/13588 [==============================] - 8s 622us/step - loss: 0.4764 - acc: 0.8728
Epoch 7/15
13588/13588 [==============================] - 8s 620us/step - loss: 0.4493 - acc: 0.8817
Epoch 8/15
13588/13588 [==============================] - 8s 624us/step - loss: 0.4243 - acc: 0.8903
Epoch 9/15
13588/13588 [==============================] - 8s 624us/step - loss: 0.3913 - acc: 0.8970
Epoch 10/15
13588/13588 [==============================] - 8s 625us/step - loss: 0.3829 - acc: 0.9012
Epoch 11/15
13588/13588 [==============================] - 8s 622us/step - loss: 0.3653 - acc: 0.9062
Epoch 12/15
13588/13588 [==============================] - 8s 621us/step - loss: 0.3579 - acc: 0.9104
Epoch 13/15
13588/13588 [==============================] - 8s 619us/step - loss: 0.3393 - acc: 0.9152
Epoch 14/15
13588/13588 [==============================] - 8s 621us/step - loss: 0.3256 - acc: 0.9169
Epoch 15/15
13588/13588 [==============================] - 8s 620us/step - loss: 0.3225 - acc: 0.9185

----
### Model evaluation

```python
Y_pred = model.predict_classes(X_test,batch_size = batch_size)
df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
print(classification_report(df_test.true, df_test.pred))
```

 -   | precision  |  recall | f1-score  | support 
-----|---------   |----------|----------|------------
0   |    0.92    |  0.81   |   0.86  |    1699
1    |   0.50   |   0.72   |   0.59   |    447
weighted avg   |    0.83   |   0.79   |   0.80   |   2146


So the class imbalance is reduced significantly recall value for positive tweets (Class 1) improved from 0.54 to 0.77. It is always not possible to reduce it completely. 
You may also noticed that the recall value for Negative tweets also decreased from 0.90 to 0.78  but this can be improved using training model to more epochs and tuning the hyper-parameters.

-------

### model inference 
```python
twt = ['keep up the good work']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=29, dtype='int32', value=0)

sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")
```

Positive

------
Try varying the class weight and run the model to bigger epoch number (100) and find best value your self.