from preprocessor import *
import os
import tensorflow as tf
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import load_model
from pickle import dump
from nltk.translate.bleu_score import corpus_bleu

#---------------------- Global variables -----------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

#---------------------- Helper functions -----------------------------------------

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
	
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		print desc_list
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input   -- check shape
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

#------------------------- Main function ---------------------------------------------


if __name__ == "__main__":
	# load the tokenizer
	tokenizer = load(open('../fT/tokenizer.pkl', 'rb'))
	# pre-define the max sequence length (from training)
	max_length = 34
	# load the model
	model = load_model('../savedModels/model1-ep012-loss3.188-val_loss3.719.h5')
	# load and prepare the photograph
	'''
	csv_f = open( '../photos/prd.csv' , 'a')
	csv_f.write('name,desc'+ '\n')
	for fname in os.listdir('../photos'):
	  if fname[-3:] =='jpg':
	    photo = extract_features('../photos/' +fname)
	    # generate description
	    description = generate_desc(model, tokenizer, photo, max_length)
	    print(description)
	    csv_f.write(fname[:-3] + ',' + description + '\n')
	csv_f.close()
	'''
	# load test set
	filename = '../Flickr8k_text/Flickr_8k.testImages.txt'
	test = load_set(filename)
	print('Dataset: %d' % len(test))
	# descriptions
	test_descriptions = load_clean_descriptions('../Flickr8k_text/descriptions.txt', test)
	print('Descriptions: test=%d' % len(test_descriptions))
	# photo features
	test_features = load_photo_features('../fT/features.pkl', test) 
	print('Photos: test=%d' % len(test_features))
	# load the model
	filename = '../savedModels/model1-ep012-loss3.188-val_loss3.719.h5'
	model = load_model(filename)
	# evaluate model
	evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
