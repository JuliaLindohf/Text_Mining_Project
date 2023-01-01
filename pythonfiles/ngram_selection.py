import numpy as np
import pandas as pd
import spacy
import json 
import nltk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras.preprocessing.text import *
from nltk.corpus import stopwords
import os
nltk.download('omw-1.4')
import spacy
import nltk
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy.displacy as displacy
from keras.preprocessing import text 
from keras.preprocessing.sequence import skipgrams
from keras.layers import Concatenate
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dot
from keras.preprocessing import text
from keras.preprocessing import sequence
import ast 
import seaborn as sns 

nltk.download('stopwords')
nltk.download('wordnet')

class TwoGrams_Extraction: 
  def __init__(self, inputlist):
    # dictionary for twograms 
    self.twogram_dict={}  
    self.reviews = inputlist
    # to store all unique two grams 
    self.uniquelist = []

  def extract_twogram(self): 
    # to load nlp package 
    nlp = spacy.load('en_core_web_sm')   

    def cleanlist(review): 
      # to split the list into words 
      listofwords = review.split() 
      listofwords2 = [word.lower() for word in listofwords if word.isalpha()] 
      words = ' '.join(listofwords2) 
      doc = nlp(words) 
      newlist = [token.lemma_ for token in doc]
      return newlist 

    for sent in self.reviews:
      cleanedlist = cleanlist(sent) 
      LW = len(cleanedlist)
      ngramlist = [(cleanedlist[l],cleanedlist[l+1])   for l in range(LW-1)]  
      for ngram in ngramlist:
        if ngram not in self.uniquelist:
          self.uniquelist.append(ngram) 
        if ngram not in self.twogram_dict:
          # to count the number of two grams 
          self.twogram_dict[ngram] = 1
        else: 
          self.twogram_dict[ngram] += 1 

   def sort_twograms(self, popularity_threshold):
      nlp = spacy.load('en_core_web_sm')   
      # to fetch the most popular noun phrases  
      newdict = {}
      for k,v in self.twogram_dict.items(): 
        if v > popularity_threshold:
          newdict[k] = v 
      self.popular_nounphrases = []
      self.popular_adjectivephrases = []
      for k,v in newdict.items(): 
        newngram = ' '.join(k)
        doc = nlp(newngram) 
        token = newngram[1]
        if token.pos_ == 'NOUN':
          self.popular_nounphrases.append(newngram) 
        if token.pos_ == 'ADJ':
          self.popular_adjectivephrases.append(newngram) 
