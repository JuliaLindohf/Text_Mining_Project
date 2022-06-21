import numpy as np
import pandas as pd
import spacy
import json 
import nltk
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
import operator
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.utils import np_utils 

# to construct an object class, to select the most popular words 
class Skimgram_Detection: 
   def __init__(self, input_textchunks):   
     # the input text chunks are cleaned text chunks 
     self.inputtextchunk = input_textchunks 

   def create_word_index(self): 
    # to use a standard Keras function tokenizer 
    tokenizer = text.Tokenizer() 
    # to tokenise the text
    tokenizer.fit_on_texts(self.inputtextchunk)  

    # to give every word an index 
    # the results are stored in a dictionary 
    self.word_to_id = tokenizer.word_index 

    self.wordindex_dict = { }
    # to create a dictionary, storing the index 
    for k,v in self.word_to_id.items(): 
      self.wordindex_dict[v] = k 
    
    # to create a new dictionary, storing the word frequencies 
    word_frec_dict = { }  
    # to create a list to store word list
    # also a list to store skipgrams 
    wordlist = []

    for chunk in self.inputtextchunk: 
      currentlist = []
      for word in chunk: 
        currentlist.append(self.word_to_id[word]) 
        # to count the number of times a word appears in the text chunks
        if word not in word_frec_dict:
          word_frec_dict[word] = 1
        else: 
          word_frec_dict[word] += 1 
      wordlist.append(currentlist)
    sortlist= sorted(word_frec_dict.items(), key=operator.itemgetter(1)) 
    sort_frec_dict = dict(sortlist)

    self.wordfrec = sort_frec_dict
    self.wordlistsindex = wordlist 

   def skipgram_pairs(self, numberoftime): 
     # to create an empty list to store popular words 
     self.popular_words = [ ]
     # numberoftime is a user given variable 
     # to collect the most frequently utilised words 
     freq = []
     newdict = { }
     for k, v in self.wordfrec.items(): 
       if v > numberoftime: 
         # to have a list of popular words 
         self.popular_words.append(k) 
         # to calculate the number of time, a word appeared 
         freq.append(v) 
         newdict[v] = k 
     words_dict = { }

     for chunk in self.inputtextchunk:
       for word in chunk:
         if word in self.popular_words:
           skipgrams = [elem for elem in chunk if elem != word] 
           if word not in words_dict: 
             words_dict[word] = [skipgrams]
           else: 
             newlist = words_dict[word]
             newlist.append(skipgrams)
             words_dict[word] = newlist  
     words_count_dict = { }
     for k,v in words_dict.items():
       newdict = { } 
       list_of_words = v
       for wordlist in list_of_words:
         for word in wordlist:
           if word not in newdict:
             newdict[word] = 1
           else:
             newdict[word] += 1
       words_count_dict[k] = newdict

     self.skipgram_count = words_count_dict 
