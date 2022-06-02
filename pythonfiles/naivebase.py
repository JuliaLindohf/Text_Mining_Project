import numpy as np
import pandas as pd
import spacy
import json 
import nltk
from spacy.lang.en import English 

class training_model_naivebase:
  # to create a dictionary which stores the likelihood of every word 
  def __init__(self, X_train, y_train): 
    self.trainingdata = X_train
    self.results = y_train 
    self.length = len(y_train) 
    # to create a new empty list, storing all words
    self.allwords = [ ]

  def cleantextdata(self):
    # to store the text data in a new list 
    newtrainingdata = []   
    nlp = spacy.load('en_core_web_sm')  
    for td in self.trainingdata: 
      doc = nlp(td) 
      sentence = [ ]
      for token in doc:
        if token.is_alpha: 
          if not token.is_stop:
            tempword = token.lemma_.lower()
            sentence.append(tempword)
            self.allwords.append(tempword)
      newtrainingdata.append(sentence)
    return newtrainingdata
     
  def frequency_count(self): 
    # to fetch the cleaned text data
    trainingtext = self.cleantextdata( )
    # to create a new storage dictionary 
    wordstorage = { }
    # to store words and result in a dictionary 
    for y, x in zip(self.results, trainingtext):
      for word in x: 
        # y is either 1 or -1 
        current_combination = (word, y) 
        if current_combination in wordstorage: 
          wordstorage[current_combination] += 1 
        else: 
          wordstorage[current_combination] = 1
    self.worddict = wordstorage 

  def uniquewords(self): 
    # to fetch the word list 
    templist = self.allwords 
    # to create a new dataframe 
    dataframe = pd.DataFrame(templist, columns =['words']) 
    # to fetch a list of unique words 
    self.uniquewords = dataframe['words'].unique().tolist() 

  def training_model(self): 
    # to calculate likelihood of every word in the dictionary  
    freq_dict = self.frequency_count( )
    wordlist = self.uniquewords( ) 

    # to create an empty dictionary, to store the likelihood of each word 
    pword = { }

    # the prior function:
    prior = 0 

    # the number of unique words 
    Lwords = len( wordlist)
    # the size of the dictionary 
    Ldict = len(freq_dict)

    nr_reviews = 0
    nr_others = 0

    # to count the number of reviews and other narratives 
    for rating in self.results: 
      if rating == '1':
        nr_reviews += 1
      else:
        nr_others +=1  
 
    # the number of text chunks 
    Lchunks = len(y_train) 
    def lookforwords(freq_dict, combination): 
      if combination in freq_dict:
        freq = freq_dict[combination]
      else: 
        freq = 0
      return freq

    for word in wordlist: 
      # two combinations should be examined 
      combination1 = (word, 1) 
      p_combination1 = lookforwords(freq_dict, combination1) 
      combination2 = (word, -1) 
      p_combination2 = lookforwords(freq_dict, combination2)  

      # to calculate the likelihood that the word is a review 
      # to add a one, to avoid zeros
      p_review = (p_combination1+1)/(nr_reviews + Ldict)
      p_other = (p_combination2+1)/(nr_others + Ldict)  

      pword[word] = np.log(p_review/p_other) 


      return prior, pword 
