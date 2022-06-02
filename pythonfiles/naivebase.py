import numpy as np
import pandas as pd
import spacy
import json 
import nltk
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# the object class is based on the lecture contents, as well as the exercises from Coursera's
# natural language processing specialisation 
# a number of variations are included however 

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
    self.frequency_count( )
    freq_dict = self.worddict 
    self.uniquewords( ) 
    wordlist = self.uniquewords

    # to create an empty dictionary, to store the likelihood of each word 
    pword = { }

    # the prior function:
    prior = 0 

    # the number of unique words 
    Lwords = len(wordlist)
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
    Lwordlist = len(wordlist)
    for i in range(Lwordlist): 
      word = wordlist[i]
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

  
# training and prediction
class Prediction_Evaluation:
  # To predict which text chunk is a book review 
  def __init__(self, x_test, y_test, prior, pword): 
    self.textchunk = x_test
    self.result = y_test 
    self.priorfunction = prior 
    self.pdictionary = pword
    # to create an empty list, to store all words
    self.allwords = [ ]

  def clearntextchunks(self): 
    # to clean text data
    # to stripe away all unnecessary digits, punctuations and other contaiminations
    newtrainingdata = []   
    nlp = spacy.load('en_core_web_sm')  
    for td in self.textchunk: 
      doc = nlp(td) 
      sentence = [ ]
      for token in doc:
        if token.is_alpha: 
          # to take away all stop words 
          if not token.is_stop:
            # all words are in lower case 
            tempword = token.lemma_.lower()
            sentence.append(tempword)
            self.allwords.append(tempword)
      newtrainingdata.append(sentence)
    return newtrainingdata

  def predictachunk(self, textchunk): 
    # to use the training model to predict 
    # the input data should be cleaned dataset 
    # the initial likelihood is the prior function 
    ptextchunk = self.priorfunction  
    for word in textchunk: 
      if word in self.pdictionary: 
        ptextchunk += self.pdictionary[word]  
    if ptextchunk > 0:
      result = 1
    else: 
      result = -1 
    return result 

  def predictlist(self):
    # to create a list to store result  
    predict_result = [ ]

    # to clean the text data 
    cleantextdata = self.clearntextchunks( ) 

    for textdata in cleantextdata: 
      newresult = self.predictachunk(textdata) 
      predict_result.append(newresult) 

    facit = self.result

    print(classification_report(facit, predict_result)) 
    print(confusion_matrix(facit, predict_result))   
  
# to split the data set, to begin with to select the test data
X_rest, X_test, y_rest, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
# to select both the training and validation test data
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.15, shuffle=True)
