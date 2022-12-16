import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('all')
import spacy
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# kmeans clusterings 
class KmeansClustering:
  def __init__(self, inputlist1, reviewlist): 
    self.inputlist = inputlist1
    # the second list contains all review sentences 
    # to store this variable as an object variable     
    self.referencelist = reviewlist
  def cluster_optimisation(self): 
    # word to vec encoding
    model = Word2Vec(self.referencelist, min_count=1) 
    wordlist = []
    newwordlist = []
    model = Word2Vec(newlist, min_count=1) 
    word_vector_list = [] 
    for word in self.inputlist:
      newwordvector = model[word]
      word_vector_list.append(newwordvector)
    # to implement a distance calculation function: 
    similarity_matrix = cosine_similarity(word_vector_list)
    # to store within-cluster sum-of-squares 
    wcss = []
    for n_clusters in range(1, 11):
      kmeans = KMeans(n_clusters=n_clusters) 
      kmeans.fit(similarity_matrix) 
      wcss.append(kmeans.inertia_) 
    plt.figure(figsize=(15, 10))  
    plt.plot(range(1, 11), wcss)
    plt.xlabel("Number of clusters", fontsize=20)
    plt.ylabel("Within-cluster sum-of-squares", fontsize=20)
    plt.grid(True)
    plt.show()
    self.transformed_textdata = similarity_matrix
  def Kmeanscluster(self, nrclusters): 
    # to cluster up the words 
    clusterprediction = KMeans(n_clusters = nrclusters) 
    # to make the prediction
    clusterprediction.fix(self.transformed_textdata)
    self.prediction_labels = clusterprediction.predict(self.transformed_textdata) 
    
    # to store data in the dictionary
    self.classdictionary = {}
    L = len(self.inputlist)
    for i in range(0, L): 
      label = self.prediction_labels[i]
      word = self.inputlist[i]  

      if label not in self.classdictionary:
        self.classdictionary[label] = [word] 
      else:
        wordlist = self.classdictionary[label] 
        wordlist.append(word) 
        self.classdictionary[label] = wordlist
