## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview
---
In this project, it uses dataset of [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and builds a classifier to classify 43 different kinds of traffic signs. Finally, the approach posted achieves **97.2%** accuracy on test set.

### Environment and Dependency

Run this code under `Python3.6` and make sure to have following the packages installed:

+ tensorflow
+ pandas
+ numpy
+ sklearn
+ imutils
+ opencv-python

### Directory and Files

#### Write-up
  To better understand what the code is doing and the ideas behind that, check [here](./WRITEUP.md).

#### model9_0.983.*
  Trained model with **98.3%** accuracy on validation set and **97.2%** accuracy on test set. Load trained model:

```Python
with tensorflow.Session() as sess:
  tensorflow.train.Saver().restore(sess, './model9_0.983')
  ...
```


### Run

Runnable jupyter notebook ([->link](./Traffic_Sign_Classifier.ipynb)) contains code.
