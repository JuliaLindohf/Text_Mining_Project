# the original code was written in jupiter notebook 
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
from tensorflow.keras import backend  
from tensorflow.keras.layers import Layer

class attention(Layer):
# this chunk of code was written with the basic algoritms which was described in: 
# https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/ 
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention, self).__init__()
        
    def build(self, input_shape):
        
        self.weight = self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.bias = self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(attention, self).build(input_shape)
        
    def call(self, x):
        
        e = backend.tanh(backend.dot(x, self.weight) + self.bias)
        a = backend.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return backend.sum(output, axis=1) 
      
 # the classification model 
model = Sequential()

model.add(Embedding(vocab_size, 128, input_length=max_length))
model.add(Conv1D(filters, kernel_size, padding = 'valid' , activation = 'relu', 
                 strides = 1 , input_shape = (max_length, embedding_dim)))
model.add(Conv1D(filters, kernel_size, padding = 'valid' , activation = 'relu', 
                 strides = 3 , input_shape = (max_length, embedding_dim)))
model.add(Dense(128, activation='relu'))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
model.add(attention( )) 
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(Dense(24, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



filters = 250
kernel_size = 5

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
num_epochs = 50
early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(trainingtextdata, traininglabel, 
                    validation_data=(validationtextdata, validationlabel), 
                    callbacks =[early_stop], epochs=num_epochs, 
                    batch_size=64, verbose=2) 


plt.figure(figsize=(10, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy']) 
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True) 
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True) 
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

ytest_pred = model.predict(testtextdata) 

ytestpred = []

for y in ytest_pred: 
  if y > 0.5:
    ytestpred.append(1)
  else:
    ytestpred.append(0)

cf_matrix = confusion_matrix(testlabel, ytestpred)
plt.figure(figsize=(10, 10)) 
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')  


# Predict probabilities for the test set
probs = model.predict(testtextdata).ravel()
fpr, tpr, thresholds = roc_curve(testlabel, probs)

# Compute the AUC score
auc = roc_auc_score(testlabel, probs)

# Plot the ROC curve
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label="ROC curve (AUC = %.2f)" % auc, color = 'brown', linewidth=3, )
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right') 
plt.grid()  
plt.show() 



# to perform transfer learning 
base_model = model

# Freeze the base model layers
for layer in base_model.layers:
  layer.trainable = False

# Add new layers for the new task
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
# Create a new model with the base model and the new layers
model2 = tf.keras.Model(base_model.input, predictions) 

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
num_epochs = 50

early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model2.fit(trainingtextdatab, traininglabelb, 
                    validation_data=(validationtextdatab, validationlabelb), 
                    callbacks =[early_stop], epochs=num_epochs, 
                    batch_size=64, verbose=2) 


ytest_pred = model.predict(testtextdatab)  

ytestpred = []

for y in ytest_pred: 
  if y > 0.5:
    ytestpred.append(1)
  else:
    ytestpred.append(0)

cf_matrix = confusion_matrix(testlabelb, ytestpred)
plt.figure(figsize=(10, 10)) 
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')  

# Predict probabilities for the test set
probs = model.predict(testtextdatab).ravel()
fpr, tpr, thresholds = roc_curve(testlabelb, probs)

# Compute the AUC score
auc = roc_auc_score(testlabelb, probs)

# Plot the ROC curve
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label="ROC curve (AUC = %.2f)" % auc, color = 'brown', linewidth=3, )
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right') 
plt.grid()  
plt.show()
