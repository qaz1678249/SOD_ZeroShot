#!/usr/bin/env python

import numpy as np

#This is for data analysis to determine what possible unseen class to be classified.

def compute_distance(a,b):
	a = a.reshape(-1)
	b = b.reshape(-1)
	return np.mean((a-b)**2)

onehot_mode = 0

a = np.load("features_attributes_labels.npz")

print(a['features'].shape)
print(a['attributes'].shape)
print(a['labels'].shape)
print(a['unseen_features'].shape)
print(a['unseen_attributes'].shape)
print(a['unseen_labels'].shape)


att_names = open("/home/max/Salient/ZeroS/attribute_data/attribute_names.txt","r")

att_num = 0
att_dict = dict()
for line in att_names:
	line=line[0:len(line)-1]
	att_dict[att_num] = line
	att_num+=1

labels = open("/home/max/Salient/ZeroS/attribute_data/class20_names.txt","r")

seen_label_dict = dict()
unseen_label_dict = dict()
seen_i = 0
unseen_i = 0
for line in labels:
	line=line[0:len(line)-1]
	if not(line=="bus" or line=="diningtable"):
		seen_label_dict[seen_i] = line
		seen_i+=1
	else:
		unseen_label_dict[unseen_i] = line
		unseen_i+=1

seen_label_num = seen_i
unseen_label_num = unseen_i


print(att_num)
print(seen_label_num)
print(unseen_label_num)

l_to_a = np.zeros((seen_label_num,66),dtype=np.float32)
l_to_a_num = np.zeros(seen_label_num,dtype=np.float32)

n = 0
for i in range(a['features'].shape[0]):
	"""
	print(i)
	if (np.amax(l_to_a[np.argmax(a['labels'][i])]) == 0):
		l_to_a[np.argmax(a['labels'][i])] = a['attributes'][i]
	else:
		if (np.allclose(l_to_a[np.argmax(a['labels'][i])] , a['attributes'][i])):
			#continue
			#print("right")
			n+=1
		else:
			#print(l_to_a[np.argmax(a['labels'][i])])
			#print(a['attributes'][i])
			#print("error")
			continue
	"""
	if (i % 100 == 0):
		print(i)
	if (onehot_mode == 1):
		ind = np.argmax(a['labels'][i])
	else:
		ind = a['labels'][i]
	l_to_a[ind] += a['attributes'][i]
	l_to_a_num[ind] += 1.0
	#if (a['attributes'][i][54] == 1 and ind == 0):
	#	print("?????????? :",i)

print(l_to_a_num)

for i in range(seen_label_num):
	l_to_a[i] = l_to_a[i] / l_to_a_num[i]
	print("$$$$$$$$$$$$$$$$$$$")
	print(i,seen_label_dict[i],np.argmax(l_to_a[i]),att_dict[np.argmax(l_to_a[i])],np.amax(l_to_a[i]))
	#print(l_to_a[i])
	all_flag = 1
	for j in range(att_num):
		if not(l_to_a[i][j]==0):
			print(att_dict[j],l_to_a[i][j])
			flag = 1
			for k in range(seen_label_num):
				if (not(i==k)):
					if not(l_to_a[k][j]==0):
						flag=0
						break
			if (flag==1):
				all_flag = 0
	if (all_flag==1):
		print("This could be detectable unseen class!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
					


maxx = -1.0
minx = 999999999.9
mini = 0
minj = 0
dist = np.zeros(seen_label_num*(seen_label_num-1)//2, dtype = np.float32)
indi = np.zeros(seen_label_num*(seen_label_num-1)//2, dtype = np.int32)
indj = np.zeros(seen_label_num*(seen_label_num-1)//2, dtype = np.int32)
haha = 0
for i in range(seen_label_num-1):
	for j in range(i+1,seen_label_num):
		dist[haha] = compute_distance(l_to_a[i],l_to_a[j])
		indi[haha] = i
		indj[haha] = j
		if (dist[haha]>maxx):
			maxx = dist[haha]
		if (dist[haha]<minx):
			minx = dist[haha]
			mini = i
			minj = j
		haha += 1

print(maxx)
print(minx)
print(mini)
print(minj)

seq = np.argsort(dist)
for i in range(seen_label_num*(seen_label_num-1)//2):
	print(dist[seq[i]],seen_label_dict[indi[seq[i]]],indi[seq[i]],seen_label_dict[indj[seq[i]]],indj[seq[i]])

