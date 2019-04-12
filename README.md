# Unconstrained Salient Object Detection and Zero-shot Classification

We use salient object detection method described in

> [Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price and Radomír Mech. "Unconstrained Salient Object Detection via Proposal Subset Optimization." CVPR, 2016.](http://cs-people.bu.edu/jmzhang/sod.html)

This method aims at producing a highly compact set of detection windows for salient objects in uncontrained images, which may or may not contain salient objects.

## Prerequisites
1. Linux
2. Matlab 
3. Caffe & Matcaffe (**We use the official master branch downloaded on 4/1/2016. Previous versions may not be compatible.**)

## Quick Start for Salient Object Detection
1. Unzip the files to a local folder (denoted as **root_folder**).
2. Enter the **root_folder** and modify the Matcaffe path in **setup.m**.
3. In Matlab, run **setup.m** and it will automatically download the pre-trained GoogleNet model.
4. Run **demo.m**.

## Miscs
To change CNN models or other configurations, please check **getParam.m**.

