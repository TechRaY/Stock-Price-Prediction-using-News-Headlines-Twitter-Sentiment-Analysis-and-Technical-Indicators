import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, Merge,BatchNormalization, Flatten, Reshape, Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()



# Need to use 300 for embedding dimensions to match GloVe's vectors.
embedding_dim = 300

nb_words = 2766
max_daily_length=200

word_embedding_matrix=pd.read_csv("input/word_embedding_matrixtcs.csv")
x_train =pd.read_csv("input/x_traintcs.csv")
x_test = pd.read_csv("input/x_testtcs.csv")
y_train =pd.read_csv("input/y_traintcs.csv")
y_test = pd.read_csv("input/y_testtcs.csv")


print("X_train shape")
print(x_train.shape)
print(len(x_train))
print(len(x_test))



filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
nb_filter = 16
rnn_output_size = 128
hidden_dims = 128
wider = False
deeper = True

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2


def build_model():
    
    model1 = Sequential()
    
    model1.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model1.add(Dropout(dropout))
    
    model1.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length1, 
                             padding = 'same',
                            activation = 'relu'))
    model1.add(Dropout(dropout))
    
    if deeper == True:
        model1.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length1, 
                                 padding = 'same',
                                activation = 'relu'))
        model1.add(Dropout(dropout))
    
    model1.add(LSTM(rnn_output_size, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout))
    
    ####

    model2 = Sequential()
    
    model2.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model2.add(Dropout(dropout))
    
    
    model2.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length2, 
                             padding = 'same',
                             activation = 'relu'))
    model2.add(Dropout(dropout))
    
    if deeper == True:
        model2.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length2, 
                                 padding = 'same',
                                 activation = 'relu'))
        model2.add(Dropout(dropout))
    
    model2.add(LSTM(rnn_output_size, 
                    activation=None,
                    kernel_initializer=weights,
                    dropout = dropout))
    
    ####

    model = Sequential()

    model.add(Merge([model1, model2], mode='concat'))
    
    model.add(Dense(hidden_dims, kernel_initializer=weights))
    model.add(Dropout(dropout))
    
    if deeper == True:
        model.add(Dense(hidden_dims//2, kernel_initializer=weights))
        model.add(Dropout(dropout))

    model.add(Dense(1, 
                    kernel_initializer = weights,
                    name='output'))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate,clipnorm=1.0))
    return model


									
# Make predictions with the best weights
deeper=True
wider=False
dropout=0.3
learning_Rate = 0.001
# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()

model.load_weights('question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout))
predictions = model.predict([x_test,x_test], verbose = True)


# In[313]:

# Compare testing loss to training and validating loss
print("mse "+str(mse(y_test, predictions)))


max_price=153.94995100000006
min_price=-255.0
# In[314]:

rk=y_test.values
#print(len(rk))
normpreds=pd.read_csv("input/predstcs.csv")
unnorm_preds=normpreds.values
#unnorm_preds=predictions

def unnormalize(price):
    price = price*(max_price-min_price)+min_price
    return(price)




unnorm_predictions = []
for pred in unnorm_preds:
    unnorm_predictions.append(unnormalize(pred))
  

unnorm_y_test = []
for y in rk:
    unnorm_y_test.append(unnormalize(y))


# Calculate the median absolute error for the predictions
print("mae "+str(mae(unnorm_y_test, unnorm_predictions)))
print("mse "+str(mse(unnorm_y_test, unnorm_predictions)))

# In[362]:

print("Summary of actual Closing price changes")
print(pd.DataFrame(unnorm_y_test, columns=[""]).describe())
print()
print("Summary of predicted Closing price changes")
print(pd.DataFrame(unnorm_predictions, columns=[""]).describe())

plt.figure(figsize=(12,4))
plt.plot(unnorm_predictions,label="Model Predictions")
plt.plot(unnorm_y_test,label="Actual Values")
plt.title("Predicted vs Actual Closing Price Changes for TCS")
plt.xlabel("Testing instances")
plt.ylabel("Change in Closing Price")
plt.legend()
plt.show()


# Create lists to measure if opening price increased or decreased
direction_pred = []
for pred in unnorm_predictions:
    if pred >= 0:
        direction_pred.append(1)
    else:
        direction_pred.append(0)
direction_test = []
for value in unnorm_y_test:
    if value >= 0:
        direction_test.append(1)
    else:
        direction_test.append(0)


		
		
# Calculate if the predicted direction matched the actual direction
direction = acc(direction_test, direction_pred)
direction = round(direction,4)*100
print("Predicted values matched the actual direction {}% of the time.".format(direction))

