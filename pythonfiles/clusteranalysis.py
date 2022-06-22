import numpy as np
import pandas as pd
import spacy
import json 
import nltk
import spacy
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
import gensim.downloader as api
from sklearn.manifold import TSNE
import operator
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.utils import np_utils 
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans 


class Cluster_Words:
  def __init__(self, inputlist, worddict, word_weights): 
    self.worddict = worddict 
    self.inputwords = inputlist
    self.wordweights = word_weights 

  def clusterwords(self): 
    wordid = [ ]
    wordvetor= [ ]
    for word in self.inputwords: 
      # to give every included word an ID
      wordid.append(self.worddict[word]) 
      wordvetor.append(self.wordweights[self.worddict[word]])
    wordvectornp = np.array(wordvetor)
    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
    np.set_printoptions(suppress=True)
    word_transformation = tsne.fit_transform(wordvectornp)
    kmeans = KMeans(n_clusters=4) 
    kmeans.fit(word_transformation) 
    Kmeanclusters = kmeans.predict(word_transformation)

    labels = self.inputwords
    plt.figure(figsize=(20, 20))
    plt.scatter(word_transformation[:, 0], word_transformation[:, 1], c= Kmeanclusters , edgecolors='k')
      
    for label, x, y in zip(labels, word_transformation[:, 0], word_transformation[:, 1]):
      plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points', fontsize=12)
    
    clusterdict = { } 
    for i in range(len(Kmeanclusters)): 
      label = Kmeanclusters[i] 
      word = self.inputwords[i] 
      if label not in clusterdict: 
        wordlist = [word]
        clusterdict[label] = wordlist
      else: 
        clusterdict[label].append(word) 
    return clusterdict

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
   def sortdict(self): 
     # I will only include words which appear at least three times 
     newwordfrec_dict = {}
     for k,v in self.wordfrec.items(): 
       if v > 1: 
         newwordfrec_dict[k] = v
     self.freq_wordfrec = newwordfrec_dict
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
    
if __name__ == '__main__':
  
# According to the lecture notes

model1 = Skimgram_Detection(ordlista)
model1.create_word_index( ) 
model1.sortdict()
model1.skipgram_pairs(1000)

# to calculate the size of the vocabulary 
size_vocabulary = len(model1.freq_wordfrec)

cbow_model = Sequential()
# input_length indicates the window size, the number of words which are considered
cbow_model.add(Embedding(input_dim=size_vocabulary , output_dim = 100, input_length=6))
# depends on the number of words
cbow_model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(100,)))
cbow_model.add(Dense(size_vocabulary, activation='softmax'))

cbow_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

weights = cbow_model.get_weights()[0]
weights = weights[1:] 


Lf = model1.popular_words 
Lf1 = ' '.join(Lf)

nlp = spacy.load("en_core_web_sm")
doc = nlp(Lf1)

adjectives = [ ]
verbs = [ ]
nouns = [ ]
for token in doc: 
  if token.pos_ == 'NOUN': 
    nouns.append(str(token))
  if token.pos_ == 'VERB':
    verbs.append(str(token))
  if token.pos_ == 'ADV':
    adjectives.append(str(token))
    
    
wordtoindexdict = {v:k for k, v in model1.wordindex_dict.items()}

model2 = Cluster_Words(adjectives, wordtoindexdict, word_vectors)
model2.clusterwords()
