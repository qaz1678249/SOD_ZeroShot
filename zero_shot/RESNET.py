#!/usr/bin/env python

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np

#this is for extracting and saving features with pre-trained RESNET50 and saveing the one-hot labels & attribiutes as well.
 
#6340 trainingdata
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

base_img_path = "/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedtrain/"

final_out_feature = []
for i in range(6340):
	img_path = base_img_path + str(i) + ".jpg"
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features_2048 = model.predict(x)
	features_2048 = features_2048.reshape(-1)

	final_out_feature.append(features_2048)
	print(i)

atts = open("/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedlabel/train_label_att.txt","r")
labels = open("/home/max/Salient/ZeroS/attribute_data/class_names.txt","r")

final_out_onehot = []
final_out_att = []
label_dict = dict()
i = 0
for line in labels:
	line=line[0:len(line)-1]
	label_dict[line] = i
	i+=1

label_num = i
i = 0
for line in atts:
	temp = line.split()
	onehot = np.zeros(label_num, dtype=np.float32)
	onehot[label_dict[temp[0]]]=1.0
	final_out_onehot.append(onehot)

	atts_i = np.zeros(64, dtype = np.float32)
	for j in range(64):
		atts_i[j] = float(int(temp[j+1]))
	final_out_att.append(atts_i)
	i+=1
	print(i)

print((np.array(final_out_onehot)).shape)
print((np.array(final_out_att)).shape)
print((np.array(final_out_feature)).shape)
np.savez("features_attributes_labels", features=np.array(final_out_feature), attributes=np.array(final_out_att), labels=final_out_onehot)
