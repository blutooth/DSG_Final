# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

import numpy as np
# load pima indians dataset
X = np.load('../Data/X.npy')
Y = X[:, 0]
X = np.array(X[:, 1::], dtype=np.int32)
X.shape
# create model
model = Sequential()
#model.add(Dropout(0.2, input_shape=(8,)))
model.add(Dense(50, input_dim=8, activation='relu', W_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
model.add(Dense(8, init='normal', activation='relu', W_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=40, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
