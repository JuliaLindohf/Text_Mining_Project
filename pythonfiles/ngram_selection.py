import numpy as np
import pandas as pd
import spacy
import json 
import nltk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras.preprocessing.text import *
from nltk.corpus import stopwords
import os
nltk.download('omw-1.4')
import spacy
import nltk
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.tokenize import sent_tokenize, word_tokenize
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
import seaborn as sns 

nltk.download('stopwords')
nltk.download('wordnet')

