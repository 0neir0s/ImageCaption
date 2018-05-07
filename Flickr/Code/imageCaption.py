from preprocessor import *
import os
import tensorflow as tf
import numpy as np
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from pickle import dump
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

#---------------------- Global variables -----------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"]="2"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
vocab_size = 7579
max_length = 34

#------------------------- Loading data --------------------------------

X1train = np.load("data/X1train.npy")
X2train = np.load("data/X2train.npy")
ytrain = np.load("data/ytrain.npy")
X1test = np.load("data/X1test.npy")
X2test = np.load("data/X2test.npy")
ytest = np.load("data/ytest.npy")
print "data loaded"

#---------------------- Helper functions -----------------------------------------

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0))
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model

#------------------- Model definition and training -------------------------------

# define the model
model = define_model(vocab_size, max_length)
model.load_weights('../savedModels/model0-ep004-loss3.620-val_loss3.854.h5')
# define checkpoint callback
filepath = '../savedModels/model1-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=30, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
