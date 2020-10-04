import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import wordcloud as wc 
import seaborn as sns 
import nltk
import re
import contractions
import symspellpy

class Preprocessor:
    def __init__(self, path_to_words_corpus):
        self.sym_spell = symspellpy.SymSpell()
        self.sym_spell.create_dictionary(path_to_words_corpus)
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        self.stopwords_corpus = set(nltk.corpus.stopwords.words())
        self.stemmer = nltk.stem.PorterStemmer()
    def preprocess(self,sentence, remove_stopwords=True, stem_reduce=True):
        sentence = sentence.lower()
        sentence = re.sub(r"\d+", "", sentence)
        sentence = contractions.fix(sentence)
        sentence = self.tokenizer.tokenize(sentence)
        if(remove_stopwords):
            sentence = [word for word in sentence if not word in self.stopwords_corpus]
        if(stem_reduce):
            sentence = [self.stemmer.stem(word) for word in sentence]
        sentence = [self.sym_spell.lookup(word, 0, include_unknown=True)[0].term for word in sentence]
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