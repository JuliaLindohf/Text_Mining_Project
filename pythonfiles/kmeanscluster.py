import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import spacy
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
