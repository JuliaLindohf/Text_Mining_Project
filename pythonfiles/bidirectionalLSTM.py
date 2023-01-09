import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import tensorflow as tf
from keras.layers import Dropout
from keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional
from tensorflow.keras.models import Sequential
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences 

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)))
model.add(Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=False)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(Dense(24, activation='relu')) 
#model.add(Dropout(drop_value)) 
model.add(Dense(1, activation='sigmoid'))

model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

filmreviews = pd.read_csv('/content/drive/MyDrive/computational_linguistics/textclassification/trainingtextdatatraining_filmreviews2.csv')
filmreviews = filmreviews.dropna()
filmreview_text = filmreviews['cleaned text'].tolist() 
filmreview_label = filmreviews['training label'].tolist() 

max_features = 2000
# the maximum number of words 
max_length = 200 
# the number of unique words in the word set 
vocab_size = 20000
# out of vocabulary token
oov_tok = '<OOV>' 
trunc_type = 'post'
padding_type = 'post'
embedding_dim = 16
drop_value = 0.2
n_dense = 24

encoding = [one_hot(words,vocab_size) for words in filmreview_text]

emb_doc = pad_sequences(encoding,
    maxlen=200,
    dtype='int32',
    padding='pre',
    truncating='pre',
    value=0.0
) 


bookreviews = pd.read_csv('/content/drive/MyDrive/computational_linguistics/textclassification/trainingtextdatatraining_bookreview.csv')
bookreviews = bookreviews.dropna() 
bookreview_text = bookreviews['training text'].tolist()
bookreview_label = bookreviews['training label'].tolist()

encoding2 = [one_hot(words,vocab_size) for words in bookreview_text ]

emb_doc2 = pad_sequences(encoding2,
    maxlen=200,
    dtype='int32',
    padding='pre',
    truncating='pre',
    value=0.0
) 

