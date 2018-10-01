
<center> <h1> Data Extraction and Text Analysis </h1> </center>

<center> <h1> Blackcoffer Consulting </h1> </center>


```python
import pandas as pd
import numpy as np
```


```python
cik_list = pd.read_excel("./cik_list.xlsx")
max_row, max_col = cik_list.shape
print(max_row)

pd.set_option('display.max_colwidth',100) # to display full text in column
cik_list.head()
```

    152
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CIK</th>
      <th>CONAME</th>
      <th>FYRMO</th>
      <th>FDATE</th>
      <th>FORM</th>
      <th>SECFNAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199803</td>
      <td>1998-03-06</td>
      <td>10-K405</td>
      <td>edgar/data/3662/0000950170-98-000413.txt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199805</td>
      <td>1998-05-15</td>
      <td>10-Q</td>
      <td>edgar/data/3662/0000950170-98-001001.txt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199808</td>
      <td>1998-08-13</td>
      <td>NT 10-Q</td>
      <td>edgar/data/3662/0000950172-98-000783.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199811</td>
      <td>1998-11-12</td>
      <td>10-K/A</td>
      <td>edgar/data/3662/0000950170-98-002145.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199811</td>
      <td>1998-11-16</td>
      <td>NT 10-Q</td>
      <td>edgar/data/3662/0000950172-98-001203.txt</td>
    </tr>
  </tbody>
</table>
</div>




```python
cik_list.SECFNAME.head()
```




    0    edgar/data/3662/0000950170-98-000413.txt
    1    edgar/data/3662/0000950170-98-001001.txt
    2    edgar/data/3662/0000950172-98-000783.txt
    3    edgar/data/3662/0000950170-98-002145.txt
    4    edgar/data/3662/0000950172-98-001203.txt
    Name: SECFNAME, dtype: object




```python
#adding the initial structure of the link in secfname
link = 'https://www.sec.gov/Archives/'
cik_list.SECFNAME = link+cik_list.SECFNAME
cik_list.SECFNAME.head()
```




    0    https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-000413.txt
    1    https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-001001.txt
    2    https://www.sec.gov/Archives/edgar/data/3662/0000950172-98-000783.txt
    3    https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-002145.txt
    4    https://www.sec.gov/Archives/edgar/data/3662/0000950172-98-001203.txt
    Name: SECFNAME, dtype: object



## Scraping data from the .txt files

For each report, we would need three sections 
- “Management's Discussion and Analysis”, 
- “Quantitative and Qualitative Disclosures about Market Risk”
- “Risk Factors”

#### The sections have specific pattern:

ITEM (section_number). section_name (*start*)

section_content (*body*)

ITEM (next_section_number) or SIGNATURES section (if section is the last one) (*end*)

#### Special case
If the form type starts with **"NT"** the theres is not data in the form so we dont need to go through them




## Text preprocessing
- Noise Removal
- Tokenization
- Normalization


```python
#varies imports

import requests
import re, string, unicodedata
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
```


```python
#making the stopword set from basic english and the given list of stopwords
nltk.download('stopwords')
nltk.download('punkt')
stopset = set(w.upper() for w in stopwords.words('english'))

```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Sanjay\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\Sanjay\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    


```python
#adding more stopwords from text file of stopwords
import glob
path = "StopWords*.txt"
glob.glob(path)
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        text = f.read()
        text = re.sub(r"\s+\|\s+[\w]*" , "", text)        
        stopset.update(text.upper().split())
        #print(len(stopset))
        

```

## In the following section a lot of useful fuctions are defined


```python
# syllables count (will be used in complex word count)
from nltk.corpus import cmudict
nltk.download('cmudict')
d = cmudict.dict()

def syllables(word):
    #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

def nsyl(word):
    try:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    except KeyError:
        #if word not found in cmudict
        return syllables(word)
```

    [nltk_data] Downloading package cmudict to
    [nltk_data]     C:\Users\Sanjay\AppData\Roaming\nltk_data...
    [nltk_data]   Package cmudict is already up-to-date!
    


```python
# other usefull functions
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_digits(text):
    return re.sub('[\d%/$]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_digits(text)
    return text

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_upper_case(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.upper()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopset:
            new_words.append(word)
    return new_words

# def stem_words(words):
#     """Stem words in list of tokenized words"""
#     stemmer = LancasterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems

# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_upper_case(words)
    words = remove_punctuation(words)
#     words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

# def stem_and_lemmatize(words):
#     stems = stem_words(words)
#     lemmas = lemmatize_verbs(words)
#     return stems, lemmas

```


```python
# section names
MDA = "Management's Discussion and Analysis"
QQDMR = "Quantitative and Qualitative Disclosures about Market Risk"
RF = "Risk Factors"
section_name = ['MDA','QQDMR',"RF"]
section = [MDA.upper(),QQDMR.upper(),RF.upper()]
variables = ['positive_score','negative_score','polarity_score','average_sentence_length', 'percentage_of_complex_words',\
                   'fog_index','complex_word_count','word_count','uncertainty_score','constraining_score', 'positive_word_proportion',\
                   'negative_word_proportion', 'uncertainty_word_proportion', 'constraining_word_proportion' ]


```


```python
import itertools

constraining_words_whole_report = pd.Series(name='constraining_words_whole_report')

df_col = [sec.lower() + '_' + var for sec,var in itertools.product(section_name,variables) ]
df = pd.DataFrame(columns=df_col)
```


```python
df.shape
```




    (0, 42)




```python
#usefull dictionaries
master_dict = pd.read_csv('./LoughranMcDonald_MasterDictionary_2016.csv', index_col= 0)
constraining_dict = set(pd.read_excel('./constraining_dictionary.xlsx',index_col = 0).index)
uncertainty_dict = set(pd.read_excel('./uncertainty_dictionary.xlsx', index_col = 0).index)
```


```python
cik_list.loc[64]
```




    CIK                                                                          4962
    CONAME                                                        AMERICAN EXPRESS CO
    FYRMO                                                                      201407
    FDATE                                                         2014-07-30 00:00:00
    FORM                                                                         10-Q
    SECFNAME    https://www.sec.gov/Archives/edgar/data/4962/0001193125-14-286961.txt
    Name: 64, dtype: object




```python
# # saving all forms locally 
# for i in range(max_row):
#     text = requests.get(cik_list.SECFNAME[i]).text
#     file_name = 'form' + str(i)
#     f = open(file_name, 'a+')
#     f.write(text)
#     f.close()
```


```python
for i in range(max_row):
    #print(i)
    file_name = './form/form' + str(i)
    text = open(file_name,'r').read()
    print('reading..',end = " ")
    
    #constraining_words_whole_report
    
#     constraining_words_whole_report_count = 0
#     for word in denoise_text(text).split():
#         if word in constraining_dict:
#             constraining_words_whole_report_count += 1
#     print('here...',end = "  ")
#     constraining_words_whole_report.loc[i] = constraining_words_whole_report_count
    
    ####################################
    df.loc[i] = np.zeros(42)
    # other variable per sections
    for j in range(3):
        if i in [63,64]:
            continue
        print(i,j,sep= '|',end = " ")
        exp = r".*(?P<start>ITEM [\d]\. " + re.escape(section[j]) + r")(?P<MDA>.*)(?P<body>[\s\S]*)(?P<end>ITEM \d|SIGNATURES)"
        regexp = re.compile(exp)
        s = regexp.search(text)
        
        if s:
            data = s.group('body')
            text = denoise_text(data)
            sent_list = sent_tokenize(text)
            sentence_length = len(sent_list)

            sample = text.split()
            sample = normalize(sample)
            word_count = len(sample)
            complex_word_count = 0
            
            for word in sample:
                if nsyl(word.lower()) > 2:
                    complex_word_count += 1
            
            average_sentence_length = word_count/sentence_length
            percentage_of_complex_words = complex_word_count/word_count
            fog_index = 0.4 * (average_sentence_length + percentage_of_complex_words)
            
            positive_score = 0
            negative_score = 0
            uncertainty_score = 0
            constraining_score = 0
            for word in sample:
                if word in master_dict.index:
                    #print("is here")
                    if master_dict.loc[word].Positive > 0:
                        #print("positive")
                        positive_score += 1
                    if master_dict.loc[word].Negative > 0:
                        negative_score += 1
                    if word in uncertainty_dict:
                        uncertainty_score += 1
                    if word in constraining_dict:
                        constraining_score += 1
            #print(positive_score)
            polarity_score = (positive_score-negative_score)/(positive_score + negative_score + .000001)
            positive_word_proportion = positive_score/word_count
            negative_word_proportion = negative_score/word_count
            uncertainty_word_proportion = uncertainty_score/word_count
            constraining_word_proportion = constraining_score/word_count
            
            df.loc[i][section_name[j].lower() + "_positive_score"] = positive_score
            df.loc[i][section_name[j].lower() + "_negative_score"] = negative_score
            df.loc[i][section_name[j].lower() + "_polarity_score"] = polarity_score
            df.loc[i][section_name[j].lower() + "_average_sentence_length"] = average_sentence_length
            df.loc[i][section_name[j].lower() + "_percentage_of_complex_words"] = percentage_of_complex_words
            df.loc[i][section_name[j].lower() + "_fog_index"] = fog_index
            df.loc[i][section_name[j].lower() + "_complex_word_count"] = complex_word_count
            df.loc[i][section_name[j].lower() + "_word_count"] = word_count
            df.loc[i][section_name[j].lower() + "_uncertainty_score"] = uncertainty_score
            df.loc[i][section_name[j].lower() + "_constraining_score"] = constraining_score
            df.loc[i][section_name[j].lower() + "_positive_word_proportion"] = positive_word_proportion
            df.loc[i][section_name[j].lower() + "_negative_word_proportion"] = negative_word_proportion
            df.loc[i][section_name[j].lower() + "_uncertainty_word_proportion"] = uncertainty_word_proportion
            df.loc[i][section_name[j].lower() + "_constraining_word_proportion"] = constraining_word_proportion        
        
            


```

    reading.. 0|0 0|1 0|2 reading.. 1|0 1|1 1|2 reading.. 2|0 2|1 2|2 reading.. 3|0 3|1 3|2 reading.. 4|0 4|1 4|2 reading.. 5|0 5|1 5|2 reading.. 6|0 6|1 6|2 reading.. 7|0 7|1 7|2 reading.. 8|0 8|1 8|2 reading.. 9|0 9|1 9|2 reading.. 10|0 10|1 10|2 reading.. 11|0 11|1 11|2 reading.. 12|0 12|1 12|2 reading.. 13|0 13|1 13|2 reading.. 14|0 14|1 14|2 reading.. 15|0 15|1 15|2 reading.. 16|0 16|1 16|2 reading.. 17|0 17|1 17|2 reading.. 18|0 18|1 18|2 reading.. 19|0 19|1 19|2 reading.. 20|0 20|1 20|2 reading.. 21|0 21|1 21|2 reading.. 22|0 22|1 22|2 reading.. 23|0 23|1 23|2 reading.. 24|0 24|1 24|2 reading.. 25|0 25|1 25|2 reading.. 26|0 26|1 26|2 reading.. 27|0 27|1 27|2 reading.. 28|0 28|1 28|2 reading.. 29|0 29|1 29|2 reading.. 30|0 30|1 30|2 reading.. 31|0 31|1 31|2 reading.. 32|0 32|1 32|2 reading.. 33|0 33|1 33|2 reading.. 34|0 34|1 34|2 reading.. 35|0 35|1 35|2 reading.. 36|0 36|1 36|2 reading.. 37|0 37|1 37|2 reading.. 38|0 38|1 38|2 reading.. 39|0 39|1 39|2 reading.. 40|0 40|1 40|2 reading.. 41|0 41|1 41|2 reading.. 42|0 42|1 42|2 reading.. 43|0 43|1 43|2 reading.. 44|0 44|1 44|2 reading.. 45|0 45|1 45|2 reading.. 46|0 46|1 46|2 reading.. 47|0 47|1 47|2 reading.. 48|0 48|1 48|2 reading.. 49|0 49|1 49|2 reading.. 50|0 50|1 50|2 reading.. 51|0 51|1 51|2 reading.. 52|0 52|1 52|2 reading.. 53|0 53|1 53|2 reading.. 54|0 54|1 54|2 reading.. 55|0 55|1 55|2 reading.. 56|0 56|1 56|2 reading.. 57|0 57|1 57|2 reading.. 58|0 58|1 58|2 reading.. 59|0 59|1 59|2 reading.. 60|0 60|1 60|2 reading.. 61|0 61|1 61|2 reading.. 62|0 62|1 62|2 reading.. reading.. reading.. 65|0 65|1 65|2 reading.. 66|0 66|1 66|2 reading.. 67|0 67|1 67|2 reading.. 68|0 68|1 68|2 reading.. 69|0 69|1 69|2 reading.. 70|0 70|1 70|2 reading.. 71|0 71|1 71|2 reading.. 72|0 72|1 72|2 reading.. 73|0 73|1 73|2 reading.. 74|0 74|1 74|2 reading.. 75|0 75|1 75|2 reading.. 76|0 76|1 76|2 reading.. 77|0 77|1 77|2 reading.. 78|0 78|1 78|2 reading.. 79|0 79|1 79|2 reading.. 80|0 80|1 80|2 reading.. 81|0 81|1 81|2 reading.. 82|0 82|1 82|2 reading.. 83|0 83|1 83|2 reading.. 84|0 84|1 84|2 reading.. 85|0 85|1 85|2 reading.. 86|0 86|1 86|2 reading.. 87|0 87|1 87|2 reading.. 88|0 88|1 88|2 reading.. 89|0 89|1 89|2 reading.. 90|0 90|1 90|2 reading.. 91|0 91|1 91|2 reading.. 92|0 92|1 92|2 reading.. 93|0 93|1 93|2 reading.. 94|0 94|1 94|2 reading.. 95|0 95|1 95|2 reading.. 96|0 96|1 96|2 reading.. 97|0 97|1 97|2 reading.. 98|0 98|1 98|2 reading.. 99|0 99|1 99|2 reading.. 100|0 100|1 100|2 reading.. 101|0 101|1 101|2 reading.. 102|0 102|1 102|2 reading.. 103|0 103|1 103|2 reading.. 104|0 104|1 104|2 reading.. 105|0 105|1 105|2 reading.. 106|0 106|1 106|2 reading.. 107|0 107|1 107|2 reading.. 108|0 108|1 108|2 reading.. 109|0 109|1 109|2 reading.. 110|0 110|1 110|2 reading.. 111|0 111|1 111|2 reading.. 112|0 112|1 112|2 reading.. 113|0 113|1 113|2 reading.. 114|0 114|1 114|2 reading.. 115|0 115|1 115|2 reading.. 116|0 116|1 116|2 reading.. 117|0 117|1 117|2 reading.. 118|0 118|1 118|2 reading.. 119|0 119|1 119|2 reading.. 120|0 120|1 120|2 reading.. 121|0 121|1 121|2 reading.. 122|0 122|1 122|2 reading.. 123|0 123|1 123|2 reading.. 124|0 124|1 124|2 reading.. 125|0 125|1 125|2 reading.. 126|0 126|1 126|2 reading.. 127|0 127|1 127|2 reading.. 128|0 128|1 128|2 reading.. 129|0 129|1 129|2 reading.. 130|0 130|1 130|2 reading.. 131|0 131|1 131|2 reading.. 132|0 132|1 132|2 reading.. 133|0 133|1 133|2 reading.. 134|0 134|1 134|2 reading.. 135|0 135|1 135|2 reading.. 136|0 136|1 136|2 reading.. 137|0 137|1 137|2 reading.. 138|0 138|1 138|2 reading.. 139|0 139|1 139|2 reading.. 140|0 140|1 140|2 reading.. 141|0 141|1 141|2 reading.. 142|0 142|1 142|2 reading.. 143|0 143|1 143|2 reading.. 144|0 144|1 144|2 reading.. 145|0 145|1 145|2 reading.. 146|0 146|1 146|2 reading.. 147|0 147|1 147|2 reading.. 148|0 148|1 148|2 reading.. 149|0 149|1 149|2 reading.. 150|0 150|1 150|2 reading.. 151|0 151|1 151|2 


```python
for i in range(max_row):
    print(i,end = " ")
    file_name = './form/form' + str(i)
    text = open(file_name,'r').read()
    print('reading..',end = " ")
    
    #constraining_words_whole_report
    constraining_words_whole_report.loc[i] = 0
    constraining_words_whole_report_count = 0
    for word in denoise_text(text).split():
        if word in constraining_dict:
            constraining_words_whole_report_count += 1
    print('here...',end = "  ")
    constraining_words_whole_report.loc[i] = constraining_words_whole_report_count
```

    0 reading.. here...  1 reading.. here...  2 reading.. here...  3 reading.. here...  4 reading.. here...  5 reading.. here...  6 reading.. here...  7 reading.. here...  8 reading.. here...  9 reading.. here...  10 reading.. here...  11 reading.. here...  12 reading.. here...  13 reading.. here...  14 reading.. here...  15 reading.. here...  16 reading.. here...  17 reading.. here...  18 reading.. here...  19 reading.. here...  20 reading.. here...  21 reading.. here...  22 reading.. here...  23 reading.. here...  24 reading.. here...  25 reading.. here...  26 reading.. here...  27 reading.. here...  28 reading.. here...  29 reading.. here...  30 reading.. here...  31 reading.. here...  32 reading.. here...  33 reading.. here...  34 reading.. here...  35 reading.. here...  36 reading.. here...  37 reading.. here...  38 reading.. here...  39 reading.. here...  40 reading.. here...  41 reading.. here...  42 reading.. here...  43 reading.. here...  44 reading.. here...  45 reading.. here...  46 reading.. here...  47 reading.. here...  48 reading.. here...  49 reading.. here...  50 reading.. here...  51 reading.. here...  52 reading.. here...  53 reading.. here...  54 reading.. here...  55 reading.. here...  56 reading.. here...  57 reading.. here...  58 reading.. here...  59 reading.. here...  60 reading.. here...  61 reading.. here...  62 reading.. here...  63 reading.. here...  64 reading.. here...  65 reading.. here...  66 reading.. here...  67 reading.. here...  68 reading.. here...  69 reading.. here...  70 reading.. here...  71 reading.. here...  72 reading.. here...  73 reading.. here...  74 reading.. here...  75 reading.. here...  76 reading.. here...  77 reading.. here...  78 reading.. here...  79 reading.. here...  80 reading.. here...  81 reading.. here...  82 reading.. here...  83 reading.. here...  84 reading.. here...  85 reading.. here...  86 reading.. here...  87 reading.. here...  88 reading.. here...  89 reading.. here...  90 reading.. here...  91 reading.. here...  92 reading.. here...  93 reading.. here...  94 reading.. here...  95 reading.. here...  96 reading.. here...  97 reading.. here...  98 reading.. here...  99 reading.. here...  100 reading.. here...  101 reading.. here...  102 reading.. here...  103 reading.. here...  104 reading.. here...  105 reading.. here...  106 reading.. here...  107 reading.. here...  108 reading.. here...  109 reading.. here...  110 reading.. here...  111 reading.. here...  112 reading.. here...  113 reading.. here...  114 reading.. here...  115 reading.. here...  116 reading.. here...  117 reading.. here...  118 reading.. here...  119 reading.. here...  120 reading.. here...  121 reading.. here...  122 reading.. here...  123 reading.. here...  124 reading.. here...  125 reading.. here...  126 reading.. here...  127 reading.. here...  128 reading.. here...  129 reading.. here...  130 reading.. here...  131 reading.. here...  132 reading.. here...  133 reading.. here...  134 reading.. here...  135 reading.. here...  136 reading.. here...  137 reading.. here...  138 reading.. here...  139 reading.. here...  140 reading.. here...  141 reading.. here...  142 reading.. here...  143 reading.. here...  144 reading.. here...  145 reading.. here...  146 reading.. here...  147 reading.. here...  148 reading.. here...  149 reading.. here...  150 reading.. here...  151 reading.. here...  


```python
# joing the files for output formate

df = pd.concat([cik_list,df,constraining_words_whole_report], axis = 1)
df.shape
```




    (152, 49)




```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CIK</th>
      <th>CONAME</th>
      <th>FYRMO</th>
      <th>FDATE</th>
      <th>FORM</th>
      <th>SECFNAME</th>
      <th>mda_positive_score</th>
      <th>mda_negative_score</th>
      <th>mda_polarity_score</th>
      <th>mda_average_sentence_length</th>
      <th>...</th>
      <th>rf_fog_index</th>
      <th>rf_complex_word_count</th>
      <th>rf_word_count</th>
      <th>rf_uncertainty_score</th>
      <th>rf_constraining_score</th>
      <th>rf_positive_word_proportion</th>
      <th>rf_negative_word_proportion</th>
      <th>rf_uncertainty_word_proportion</th>
      <th>rf_constraining_word_proportion</th>
      <th>constraining_words_whole_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199803</td>
      <td>1998-03-06</td>
      <td>10-K405</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-000413.txt</td>
      <td>23.0</td>
      <td>78.0</td>
      <td>-0.544554</td>
      <td>8.920502</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199805</td>
      <td>1998-05-15</td>
      <td>10-Q</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-001001.txt</td>
      <td>11.0</td>
      <td>68.0</td>
      <td>-0.721519</td>
      <td>11.352000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199808</td>
      <td>1998-08-13</td>
      <td>NT 10-Q</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950172-98-000783.txt</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199811</td>
      <td>1998-11-12</td>
      <td>10-K/A</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-002145.txt</td>
      <td>43.0</td>
      <td>153.0</td>
      <td>-0.561224</td>
      <td>9.486076</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199811</td>
      <td>1998-11-16</td>
      <td>NT 10-Q</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950172-98-001203.txt</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199811</td>
      <td>1998-11-25</td>
      <td>10-Q/A</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-002278.txt</td>
      <td>25.0</td>
      <td>181.0</td>
      <td>-0.757282</td>
      <td>10.500000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199812</td>
      <td>1998-12-22</td>
      <td>10-Q</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-002401.txt</td>
      <td>44.0</td>
      <td>147.0</td>
      <td>-0.539267</td>
      <td>9.096859</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199812</td>
      <td>1998-12-22</td>
      <td>10-Q</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-002402.txt</td>
      <td>41.0</td>
      <td>126.0</td>
      <td>-0.508982</td>
      <td>9.452381</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199903</td>
      <td>1999-03-31</td>
      <td>NT 10-K</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950172-99-000362.txt</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>199905</td>
      <td>1999-05-11</td>
      <td>10-K</td>
      <td>https://www.sec.gov/Archives/edgar/data/3662/0000950170-99-000775.txt</td>
      <td>84.0</td>
      <td>356.0</td>
      <td>-0.618182</td>
      <td>8.946860</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 49 columns</p>
</div>




```python
writer = pd.ExcelWriter('./output.xlsx')
df.to_excel(writer, sheet_name='output')
```
