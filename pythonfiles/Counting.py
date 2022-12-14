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
  def __init__(self, inputframe): 
    self.originaldf = inputframe
    self.reviewlist= inputframe['Review_Text'].tolist()
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
    # to create a new dictionary, to store the 2 grams 
    twogram = {} 

    def ngrams(inputlist, n, dictngram): 
        # to collect ngrams 
        L = len(inputlist)
        # to fetch ngrams from the input list 
        if n ==2:
          ngrams =[ (inputlist[i], inputlist[i+1]) for i in range(L-n+1)]  
        if n == 3: 
          ngrams =[ (inputlist[i], inputlist[i+1], inputlist[i+2]) for i in range(L-n+1)]  
        if n == 4: 
          ngrams =[ (inputlist[i], inputlist[i+1], inputlist[i+2], inputlist[i+3]) for i in range(L-n+1)]  
        # to store ngrams in the list 
        for detectedngrams in ngrams:
          if detectedngrams in dictngram: 
            dictngram[detectedngrams] += 1
          else: 
            dictngram[detectedngrams] = 1

        return dictngram 

    for review in reviewlist: 
      doc = nlp(review) 
      # to create an empty list to store cleaned text data 
      texttokens = [ ] 
      for word in doc:
        if word.is_alpha:
          word2 = word.lemma_.lower() 
          if word2 == '-pron-':
            continue  
          texttokens.append(word2)
          if word2 in self.worddict: 
            self.worddict[word2] += 1
          else: 
            self.worddict[word2] = 1
      self.twograms = ngrams(texttokens, 2, self.twograms) 
      self.threegrams = ngrams(texttokens, 3, self.threegrams)  
      self.fourgrams = ngrams(texttokens, 4, self.fourgrams)  

  def Entity_survey(self): 
    reviewlist = self.reviewlist 
    nlp = spacy.load('en_core_web_sm')  
    for review in reviewlist: 
      doc = nlp(review)

      # to extract entities 
      for entity in doc.ents: 
        # to take out the labels 
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
    
class Estimating_N_Grams: 
  def __init__(self, inputdataframe): 
    self.dataframe = inputdataframe 
    self.bookreview = inputdataframe['book review'] 
    self.rating = inputdataframe['rating'] 
    self.wordcount = inputdataframe['number of words'] 

  def cleaningdata(self, rating): 
    nlp = spacy.load('en_core_web_sm')  

    def cleanData(doc, stemming = False):
      # to include a build function to do text data cleaning 
      # to check if the words are stop words 
      tokens = [tokens for tokens in doc if (tokens.is_stop == False)]
      tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]
      # to find the stem of the words 
      final_token = [token.lemma_ for token in tokens]
      # to put the words together into a pharaphrase again 
      return " ".join(final_token)

    # to count the number of element 
    L = len(self.rating)
    # to store the texts in a list
    textcollectionlist = []

    for i in range(0, L): 
      currentrating = self.rating[i] 
      if currentrating == rating:
        textcollectionlist = textcollectionlist + self.bookreview[i].split()
    # to clearn the entire data 
    # the intention is to use the entire corpus for text mining later 
    newtest = ' '.join(textcollectionlist) 
    self.cleaned_text = cleanData(newtest)

  def find_twograms(self, rating): 
    # to fetch the cleaned data 
    cleaneddata = self.cleaningdata(rating) 
    # to save the results in a dictionary 
    twogram_dict = {} 
    twogram_set = set()
    for list1 in cleaneddata:
      list2 = list1.split() 
      ngramlist = [(list2[l], list2[l+1])   for l in range(0, len(list2)-1)] 
      for ngram in ngramlist: 
        if ngram not in twogram_dict:
          twogram_dict[ngram] = 1
          twogram_set.add(ngram)
        else: 
          twogram_dict[ngram] += 1
    return twogram_dict, twogram_set 


  def find_threegrams(self, rating): 
    # to fetch the cleaned data 
    cleaneddata = self.cleaningdata(rating) 
    # to save the results in a dictionary 
    threegram_dict = {} 
    threegram_set = set()
    for list1 in cleaneddata:
      list2 = list1.split() 
      ngramlist = [(list2[l], list2[l+1], list2[l+2]) for l in range(0, len(list2)-2)]  
      for ngram in ngramlist: 
        if ngram not in threegram_dict:
          threegram_dict[ngram] = 1
          threeram_set.add(ngram)
        else: 
          threegram_dict[ngram] += 1
    return threegram_dict, threegram_set 
