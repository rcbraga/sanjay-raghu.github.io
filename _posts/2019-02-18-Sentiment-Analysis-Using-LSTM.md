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

