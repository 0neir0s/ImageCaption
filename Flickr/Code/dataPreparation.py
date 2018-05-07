from preprocessor import *
import os
import tensorflow as tf
from pickle import load
from numpy import argmax
from pickle import dump
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#---------------------- Global variables -----------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

#--------------------------- Main ------------------------------------------------

# load training dataset (6K)
filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('../Flickr8k_text/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('../fT/features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('../fT/tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features,vocab_size)
# load test set
filename = '../Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('../Flickr8k_text/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('../fT/features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features,vocab_size)
exit(0)

#------------------- Saving the data ----------------------------------------------------------------------

np.save("data/X1train.npy", X1train)
np.save("data/X2train.npy", X2train)
np.save("data/ytrain.npy", ytrain)
np.save("data/X1test.npy", X1test)
np.save("data/X2test.npy", X2test)
np.save("data/ytest.npy", ytest)
