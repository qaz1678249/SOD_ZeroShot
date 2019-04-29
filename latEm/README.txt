This is Matlab implementation for latEm model described in  
Y. Xian, Z. Akata, G. Sharma, Q. Nguyen, M. Hein, B. Schiele. 
Latent Embeddings for Zero-shot Classification. IEEE CVPR 2016.
Cite the above paper if you are using this code.


'latEm_train' Usage
=================

Usage: W = latEm_train(X, labels, Y, eta, n_epoch, K)
Inputs:
-X:                  images embedding matrix, each row is an image instance
-Y:                  class embedding matrix, each col is for a class
-labels:             ground truth labels of all image instances
-eta:                learning rate for SGD algorithm
-n_epoch:            number of epochs for SGD algorithm
-K:                  number of embeddings to learn

Outputs:
-W:                  a cell array with K embeddings


'latEm_test' Usage
=================

Usage: [mean_class_accuracy] = latEm_test(W, X, Y, labels)

Inputs:
-W:                  latent embeddings
-X:                  images embedding matrix, each row is an image instance
-Y:                  class embedding matrix, each col is for a class
-labels:             ground truth labels of all image instances

Outputs:
-mean_class_accuracy: the classification accuracy averaged over all classes


Data
=================

In the zero-shot setting, we split each dataset into train and test set which 
have disjoint classes. In order to pick the hyperparameters i.e. learning rate
and number of latent embeddings, we further split the train set into a smaller 
train set and a validation set which have disjoint classes. The final model is
trained on train+validation set. 

We released the data for CUB dataset. 'data_CUB.mat' includes the following fields:

-trainval_X:         images embedding matrix(cnn feature obtained from googLeNet) of train+validation set, each row is an image instance
-train_X:            ...
-val_X:              ...
-test_X:             ...

-trainval_Y          a hash map with the class embedding name as the key, and its class embedding matrix as the value. 
                     eg. trainval_Y('word2vec') is the class embedding matrix of train+validation set, each col is for a class
-train_Y             ...
-val_Y               ...
-test_Y              ...
-cls_emb_names:      names for the available class embeddings in the hash map, 'word2vec' and 'glove' are discribed in the paper, 
                    'wordnet' corresponds to hierarchical embedding in the paper, and 'cont' corresponds to continuous attribute 
                    in the paper

-trainval_labels:    ground truth labels of all image instances
-train_labels:       ...
-val_labels:         ...
-test_labels:        ...


Demo
=================

demo_main.m shows how to use the data_CUB.mat, latEm_train and latEm_test


CONTACT:
=================
Yongqin Xian
e-mail: yxian@mpi-inf.mpg.de
Computer Vision and Multimodal Computing, Max Planck Institute Informatics
Saarbruecken, Germany
http://d2.mpi-inf.mpg.de
