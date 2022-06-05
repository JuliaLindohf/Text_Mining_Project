import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
import json 
import nltk
nltk.download('punkt')
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns 


class phrase_freq:
  def __init__(self, inputframe, column): 
    self.originaldf = inputframe
    self.reviewlist= inputframe[column].tolist()
    self.cleanedtext = [ ]
    # to create an empty dictionary to store words 
    self.worddict= { } 
    # to create an empty dictionary for n-grams 
    self.twograms = { } 
    self.threegrams = { } 
    self.fourgrams = { } 
    self.entitysurvey = { } 

  def Wordcounts(self):  
    reviewlist = self.reviewlist 
    nlp = spacy.load('en_core_web_sm') 

    def ngrams(inputlist, n, dictngram): 
        # to collect ngrams 
        L = len(inputlist)
        # to fetch ngrams from the input list 
        ngrams = [inputlist[i:i+n] for i in range(L-n+1)] 

        # to store ngrams in the list 
        for detectedngrams in ngrams:
          if detectedngrams not in dictngram: 
            dictngram[ detectedngrams ] = 1
          else: 
            dictngram[ detectedngrams ] += 1

        return dictngram 

    for review in reviewlist: 
      doc = nlp(review) 
      # to create an empty list to store cleaned text data 
      texttokens = [ ] 
      for word in doc:
        if word.is_alpha:
          word2 = word.lemma_.lower() 
          texttokens.append(word2)
          if word2 in self.worddict: 
            self.worddict[word2] += 1
          else: 
            self.worddict[word2] = 1
      
      self.twograms = ngrams(self.reviewlist, 2, self.twograms) 
      self.threegrams = ngrams(self.reviewlist, 3, self.threegrams) 
      self.fourgrams = ngrams(self.reviewlist, 4, self.fourgrams)  

    return self.worddict 
  def Entity_survey(self): 
    reviewlist = self.reviewlist 
    nlp = spacy.load('en_core_web_sm')  
    for review in reviewlist: 
      doc = nlp(review)

      # to extract entities 
      for entity in doc.ents: 
        temp_label = entity.label_
        temp_text = entity.text  
        if temp_label not in self.entitysurvey: 
          textdict = {}
          textdict[temp_text] = 1
          self.entitysurvey[temp_label] = textdict 
        else: 
          dictentity = self.entitysurvey[temp_label]
          if temp_text not in dictentity: 
            dictentity[temp_text] = 1
          else:
            dictentity[temp_text] += 1
          self.entitysurvey[temp_label] = dictentity
      return self.entitysurvey 
