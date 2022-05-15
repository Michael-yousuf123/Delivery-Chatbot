from nltk.tokenize import LineTokenizer
from nltk import PorterStemmer
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
bag_of_words_model = CountVectorizer()
linetoken = LineTokenizer()
port = PorterStemmer()

def tokenization(self):
        return linetoken.tokenize(self)
def normalize(self, data):
    data = data.apply(lambda a: " " .join(a.lower() for a in a.split()))
    data = data.apply(lambda a: " " .join(a.replace('[^\w\s]','') for a in a.split()))
    data = sorted(set(data))
    return list(data)
def stem(token):
    """function that take the token 
    sentence as an argument and return
    the stemmer text
    """
    return port.stem(token)
def bag_of_words(self, corpus):
        dense_matrix = bag_of_words_model.fit_transform(corpus).todense()
        bow_df = pd.DataFrame(dense_matrix)
        bow_df.columns = sorted(bag_of_words_model.vocabulary_)
        bow = bow_df.to_numpy()
        return bow