# Salient Object Detection and Zero-shot Classification

We use salient object detection method described in

> [Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price and Radomír Mech. "Unconstrained Salient Object Detection via Proposal Subset Optimization." CVPR, 2016.](http://cs-people.bu.edu/jmzhang/sod.html)

This method aims at producing a highly compact set of detection windows for salient objects in uncontrained images, which may or may not contain salient objects.
And then we are going to use zero-shot learning to classify seen and unseen objects.

## Prerequisites
1. Linux
2. Matlab 
3. Caffe & Matcaffe (**Previous than 4/1/2016 versions may not be compatible.**)

## Dataset
We are using aPASCAL VOC2008(VOC with attributes). This dataset is provided by

> [A. Farhadi, I. Endres, D. Hoiem, and D.A. Forsyth, “Describing Objects by their Attributes”, CVPR 2009.](http://vision.cs.uiuc.edu/attributes/)

For convinence, we have uploaded the dataset on Google Cloud. Here is the [Link](https://drive.google.com/drive/folders/1-QkiCcAzchzSQrUgHbspvLUK0zjbpfia?usp=sharing)

## Quick Start for Salient Object Detection
1. Unzip the files to a local folder (denoted as **root_folder**).
2. Enter the **root_folder** and modify the Matcaffe path in **setup.m**.
3. In Matlab, run **setup.m** and it will automatically download the pre-trained GoogleNet model.
4. Run **demo.m**.
To change CNN models or other configurations, please check **getParam.m**.

