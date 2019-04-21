#!/usr/bin/env python

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np

#this is for extracting features with pre-trained RESNET50

#if 1 labels are onehot
onehot_mode = 0

base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

base_img_path = "/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedtrain/"
atts = open("/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedlabel/train_label_att.txt","r")
labels = open("/home/max/Salient/ZeroS/attribute_data/class20_names.txt","r")

final_out_onehot = []
final_out_att = []
final_out_feature = []

final_out_unseen_onehot = []
final_out_unseen_att = []
final_out_unseen_feature = []


seen_label_dict = dict()
unseen_label_dict = dict()
seen_i = 0
unseen_i = 0
for line in labels:
	line=line[0:len(line)-1]
	if not(line=="bus" or line=="diningtable"):
		seen_label_dict[line] = seen_i
		seen_i+=1
	else:
		unseen_label_dict[line] = unseen_i
		unseen_i+=1

seen_label_num = seen_i
unseen_label_num = unseen_i
i = 0

for line in atts:
	temp = line.split()

	atts_i = np.zeros(66, dtype = np.float32)
	for j in range(64):
		atts_i[j] = float(int(temp[j+1]))
	if (temp[0]=="cat"):
		atts_i[64] = 1.0
	if (temp[0]=="dog"):
		atts_i[65] = 1.0


	img_path = base_img_path + str(i) + ".jpg"
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features_2048 = model.predict(x)
	features_2048 = features_2048.reshape(-1)


	if not(temp[0]=="bus" or temp[0]=="diningtable"):
		if (onehot_mode == 1):
			onehot = np.zeros(seen_label_num, dtype=np.float32)
			onehot[seen_label_dict[temp[0]]]=1.0
		else:
			onehot = seen_label_dict[temp[0]]

		final_out_onehot.append(onehot)
		final_out_att.append(atts_i)
		final_out_feature.append(features_2048)
	else:
		if (onehot_mode == 1):
			onehot = np.zeros(unseen_label_num, dtype=np.float32)
			onehot[unseen_label_dict[temp[0]]]=1.0
		else:
			onehot = unseen_label_dict[temp[0]]

		final_out_unseen_onehot.append(onehot)
		final_out_unseen_att.append(atts_i)
		final_out_unseen_feature.append(features_2048)

	i+=1
	print(i)

print((np.array(final_out_onehot)).shape)
print((np.array(final_out_att)).shape)
print((np.array(final_out_feature)).shape)
print((np.array(final_out_unseen_onehot)).shape)
print((np.array(final_out_unseen_att)).shape)
print((np.array(final_out_unseen_feature)).shape)
np.savez("features_attributes_labels", features = np.array(final_out_feature), 
					attributes = np.array(final_out_att), 
					labels = np.array(final_out_onehot),
					unseen_features = np.array(final_out_unseen_feature),
					unseen_attributes = np.array(final_out_unseen_att),
					unseen_labels = np.array(final_out_unseen_onehot) )
