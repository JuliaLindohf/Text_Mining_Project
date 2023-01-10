import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import tensorflow as tf
from keras.layers import Dropout
from keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


# the classification model 
filters = 250
kernel_size = 5

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(Conv1D(filters, kernel_size, padding = 'valid' , activation = 'relu', 
                 strides = 1 , input_shape = (max_length, embedding_dim)))
model.add(Conv1D(filters, kernel_size, padding = 'valid' , activation = 'relu', 
                 strides = 3 , input_shape = (max_length, embedding_dim)))
model.add(GlobalMaxPooling1D())
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
num_epochs = 50
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(trainingtextdata, traininglabel, 
                    validation_data=(validationtextdata, validationlabel), 
                    callbacks =[early_stop], epochs=num_epochs, 
                    batch_size=64, verbose=2) 

