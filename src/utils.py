from nltk.tokenize import LineTokenizer
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class Preprocess:
    def __init__ (self):
        pass
    def tokenization(self):
        linetoken = LineTokenizer()
        return linetoken.tokenize(self)
    def stemming(token):
        """function that take the token 
        sentence as an argument and return
        the stemmer text
        """
        port = PorterStemmer()
        tokeStemm = [port.stem(t) for t in token]
        return tokeStemm
    def bag_of_words(self):
        """function to return the bag
        of words or vectorizer dataframe 
        from the corpus
        ================================
        ARGUMENT: corpus or sentence
        ================================
        RETURN: vectorized dataframe"""
        bag_of_words_model = CountVectorizer()
        dense_matrix = bag_of_words_model.fit_transform(self).todense()
        bow_df = pd.DataFrame(dense_matrix)
        bow_df.columns = sorted(bag_of_words_model.vocabulary_)
        return bow_df