'''
2020_09_28 (28 Sep):
Preprocessing and embedding.

Prerequisites:
nltk 
contractions
numpy
pandas (I think)
a suitable .txt of GloVe vectors. See http://nlp.stanford.edu/data/glove.6B.zip


Preprocessor:
    Methods:

    __init__(self):
        Creates tokenizer, corpus, and lemmatizer
    preprocessor(self, sentence):

        Arguments: sentence: A bog-standard Python string.

        lowercase
        contractions fix 
        tokenize (regex r"\w+")
        remove stop words
        lemmatize (wordnetlemmatizer)

        Returns: A list of strings as given above.

Glove_Embedder:
    __init__(self, PATH_TO_TEXTFILE):
        Sets up the dictionary.
    get_embedding_for_sentence(sentence):
        Arguments: sentence: A list of words. Ideally preprocessed by above.
        Returns: A list of floats.
        
        An average of the embeddings for all the words in the sentence.
        If there is no matching vector for a word, that word is given the vector 0.


    get_embeddings_for_word(word):
        Arguments: word: A string
        Returns: A list of floats, as in get_embeddings_for_sentence.
'''

import nltk
import contractions
import numpy as np 
import pandas as pd
class Preprocessor:
    def __init__(self): 
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        self.stopwords_corpus = set(nltk.corpus.stopwords.words())
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    def preprocess(self,sentence):
        sentence = sentence.lower()
        sentence = contractions.fix(sentence)
        sentence = self.tokenizer.tokenize(sentence)
        sentence = [word for word in sentence if not word in self.stopwords_corpus]
        sentence = sentence = [self.lemmatizer.lemmatize(word) for word in sentence]
        return sentence


class Glove_Embedder:
    def __init__(self, PATH_TO_TEXTFILE):
        self.glove_embeddings_dict = {}
        glove_embeddings_file = open(PATH_TO_TEXTFILE, 'r')
        firstTime = True
        while True:
            line = glove_embeddings_file.readline()
            if not line:
                break
            splitted = line.split()
            key = splitted[0]
            value = np.array([float(i) for i in splitted[1:]])
            if(firstTime):
                firstTime = False 
                self.embedding_vector_size = value.size
            self.glove_embeddings_dict[key] = value
        glove_embeddings_file.close()
    def get_embedding_for_sentence(self, sentence_list):
        '''
        The sentence should be lowercased and free of special characters and numbers. Ideally, it should be lemmatized, too. The sentence should be a list of words.
        '''
        number_of_words = len(sentence_list)
        embedding = np.zeros((self.embedding_vector_size, ))
        if(number_of_words == 0):
            return embedding 
        for word in sentence_list:
            if word in self.glove_embeddings_dict:
                embedding += self.glove_embeddings_dict[word]
        embedding /= number_of_words
        return embedding.tolist()
    def get_embedding_for_word(self, word):
        if word in self.glove_embeddings_dict:
            embedding = self.glove_embeddings_dict[word]
        else:
            embedding = np.zeros((self.embedding_vector_size, ))
        return embedding.tolist()
