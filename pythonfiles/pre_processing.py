import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English 
import json 

# the code was given by http://snap.stanford.edu/data/ 
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz
!gzip -d reviews_Kindle_Store_5.json

!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store.json.gz
!gzip -d reviews_Kindle_Store.json

!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV.json.gz
!gzip -d reviews_Movies_and_TV.json 

# to create a dataframe which stores all Kindle reviews 
f1 = open("/content/reviews_Kindle_Store_5.json", "r")
productsData1 = [json.loads(line) for line in f1.readlines()]

f2 = open("/content/reviews_Kindle_Store.json", "r")
productsData2 = [json.loads(line) for line in f2.readlines()] 

# to merge two dataframes into one. 

# to create a new dataframe, two categories, the sentiments and the reviews 
Book_Sentiment = [ ] 
Book_Reviews = [ ]  
Review_summary = [ ]

for product in productsData1: #for each line in the json file
    #if the review is empty, ignore that line
    if product["reviewText"] == "":  
        continue
    # to append one more reviews in the list
    Book_Reviews.append(product["reviewText"]) 
    Review_summary.append(product["summary"] )

    
    if product["overall"] > 3:
        Book_Sentiment.append("positive")
    else:
        Book_Sentiment.append("negative")
        
for product in productsData2: #for each line in the json file
    #if the review is empty, ignore that line
    if product["reviewText"] == "":  
        continue
    # to append one more reviews in the list
    Book_Reviews.append(product["reviewText"]) 
    Review_summary.append(product["summary"] )

    if product["overall"] > 3:
        Book_Sentiment.append("positive")
    else:
        Book_Sentiment.append("negative")

# a new panda dataframe 
datalist = {'Reader_Summary': Review_summary, 'Review_Text': Book_Reviews, 'Sentiments': Book_Sentiment}
Booksentiment_ratings = pd.DataFrame(datalist) 

# to divide the dataframe into two separate dataframes for futural analysis 
# the positive reviews 
dataframe_positive = Booksentiment_ratings[Booksentiment_ratings['Sentiments']=='positive']

# the negative reviews 
dataframe_negative = Booksentiment_ratings[Booksentiment_ratings['Sentiments']!='positive'] 
