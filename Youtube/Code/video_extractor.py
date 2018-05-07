'''
According to the paper, the authors extracted upto 80 frames from each video,
they did not mention if they grabbed first 80 frames, or sampled 80 frames with same intervals,
but anyway I did the latter.
'''
import cv2
import os
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

# extract features from each photo in the directory
def extract_features(model,frames):
    # load the photo
    feature = []
    for frame in frames:
        image = img_to_array(frame)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature.append(model.predict(image, verbose=0))
    return feature

def preprocess_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]
    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]
    return cv2.resize(resized_image, (target_height, target_width))

def main():
    num_frames = 80
    video_path = '../Videos'
    video_save_path = '../Feats'
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('avi'), videos)
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    for video in videos:
        with open('files.txt','wb') as fp:
            fp.write(video+'\n')
        print video
        if os.path.exists( os.path.join(video_save_path, video) ):
            print "Already processed ... "
            continue
        video_fullpath = os.path.join(video_path, video)
        try:
            cap  = cv2.VideoCapture( video_fullpath )
        except:
            pass
        frame_count = 0
        frame_list = []
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame_list.append(frame)
            frame_count += 1
        frame_list = np.array(frame_list)
        if frame_count > 80:
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]
        cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
        feats = extract_features(model,cropped_frame_list)
        save_full_path = os.path.join(video_save_path, video + '.npy')
        np.save(save_full_path , feats)

if __name__=="__main__":
    main()
