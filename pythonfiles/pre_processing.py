import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English 
import json 
import pyspark


# the code was given by http://snap.stanford.edu/data/ 
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz
!gzip -d reviews_Kindle_Store_5.json

!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store.json.gz
!gzip -d reviews_Kindle_Store.json

!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_10.json.gz
!gzip -d 	reviews_Movies_and_TV_10.json.gz

# to create the first dataframe which stores all Kindle reviews 
f1 = open("/content/reviews_Kindle_Store_5.json", "r")
productsData1 = [json.loads(line) for line in f1.readlines()]

# to create the second dataframe which stores all Kindle reviews 
f2 = open("/content/reviews_Kindle_Store.json", "r")
productsData2 = [json.loads(line) for line in f2.readlines()]  

# to merge two dataframes into one. 

# to create a new dataframe, two categories, the sentiments and the reviews 
Book_Reviews = [ ]  
reviewresults = [ ]

for product in productsData1: #for each line in the json file
    #if the review is empty, ignore that line
    if product["reviewText"] == "":  
        continue
    Book_Reviews.append(product["reviewText"]) 
    reviewresults.append('bookreview')
    
# Financial news     
financialdata3 = pd.read_csv('/content/drive/MyDrive/text_mining_project/financialdata/reuters_headlines.csv')

list1 = financialdata3['Headlines'].tolist( )
list2 = financialdata3['Description'].tolist( )
L = len(list1)
newlist = []

for i in range(L): 
  templist = [ ]
  temp1 = list1[i].split()
  for words in temp1:
    if words.isalpha():
      templist.append(words.lower())

  temp2 = list2[i].split()
  for words in temp2: 
    if words.isalpha():
      templist.append(words.lower())
  Book_Reviews.append(" ".join(templist))
  reviewresults.append('others')
  
# a new panda dataframe 
datalist = {'Review_Text': Book_Reviews, 'results': reviewresults}
# a new dataframe 
Booksentiment_ratings = pd.DataFrame(datalist) 
