#!/usr/bin/env python

import numpy as np
#import tensorflow
#import keras
import cv2

#this is for cropping PASCAL VOC images for the training data of zero-shot learning.

g_img_loc = "/home/max/Salient/ZeroS/VOCtrain/VOC2008/JPEGImages/"

atts = open("/home/max/Salient/ZeroS/attribute_data/apascal_train.txt","r")
l_ats = open("/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedlabel/train_label_att.txt","w")

i = 0
for line in atts:
	temp = line.split()
	jpg_name = temp[0]
	lu0 = int(temp[2])
	lu1 = int(temp[3])
	rd0 = int(temp[4])
	rd1 = int(temp[5])
	write_info = temp[1] + line[len(line)-129:len(line)]
	img_loc = g_img_loc + jpg_name
	img = cv2.imread(img_loc)
	crop_img = img[lu1:rd1+1,lu0:rd0+1]
	cv2.imwrite("/home/max/Salient/ZeroS/VOCtrain/VOC2008/croppedtrain/"+str(i)+".jpg", crop_img)
	l_ats.write(write_info)
	i+=1
	print(i)

