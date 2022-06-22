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
