#!/usr/bin/env python

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np

#this is for extracting features with pre-trained RESNET50

#6340 trainingdata
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

img_path = "/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedtrain/14.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features_2048 = model.predict(x)

print(features_2048.shape)
