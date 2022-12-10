import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
!gzip -d reviews_Movies_and_TV_5.json.gz


f = open("reviews_Movies_and_TV_5.json", "r")
filmData = [json.loads(line) for line in f.readlines()]
filmreviews = []
rating = []   
