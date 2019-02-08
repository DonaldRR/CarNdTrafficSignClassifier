# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[classes_distribution]: ./write_up_img/classes_distribution.png
[preprocess_img]: ./write_up_img/preprocess.png
[test_images]: ./write_up_img/test_images.png
[train_images]: ./write_up_img/train_images.png
[resize_test_images]: ./write_up_img/resize_test_images.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

Here is a link to my [project code](https://github.com/DonaldRR/CarNdTrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb) and this one is [README.md](https://github.com/DonaldRR/CarNdTrafficSignClassifier/blob/master/README.md).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows how classes are distributed.

![alt text][classes_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

In the preprocessing step, I add a gray-scale channel to each images and generate some rotated images to the dataset.

The following images from left to right present: original image, rotated image, gray-scale original image, gray-scale rotated image.

Then the first 3-channels image combines with the third one and we have a 4-channel "image", likewise the second one and the forth one.

![alt_text][preprocess_img]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        					                 | 
|:-------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x4   							                             | 
| Batch normalization |                                               |
| Convolution 3x3     | 1x1 stride, valid padding, outputs 30x30x16 	 |
| RELU					           |				                                    							|
| Convolution 3x3     | 1x1 stride, valid padding, outputs 28x28x32  	|
| RELU					           |				                                   								|
| Convolution 3x3     | 1x1 stride, valid padding, outputs 26x26x64  	|
| RELU					           |						                                   						|
| Convolution 5x5     | 1x1 stride, valid padding, outputs 22x22x64 	 |
| RELU					           |		                                    									|
| Convolution 5x5     | 2x2 stride, valid padding, outputs 9x9x64    	|
| RELU				           	|	                                   											|
| Flatten             | output 5184                                   |
| Fully connected		   | output 200                                   	|
| Tanh                |                                               |
| Dropout             | keep probability 0.5                          |
| Fully connected		   | output 100                                   	|
| Tanh                |                                               |
| Dropout             | keep probability 0.5                          |
|	Output				         	|	outputs 43	                         										|
 

#### 3. Describe how you trained your model. 

To train this model, I choose `softmax_cross_entropy_with_logits` as loss measurement. The optimizer is `AdamOptimizer`. 

Batch size is `128` and number of training epochs is `30`. While training model for 30 epochs may make it overfitting. So it dynamically save model in the training process: model with higher validation accuracy than the previous one will be saved.

As for the learning rate, it uses learning rate decay method: learning rate's initial value is `0.001` and it decays 5% after each epoch.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of **99.8%**
* validation set accuracy of **98.3%**
* test set accuracy of **97.2%**

In this architecture, it uses 5 convolutional layers and no pooling layer. 

Before feeding to convolutional layers, input data is normallized through batch normalization layer to make the model more easy to train.

The firt three convolutional layers has kernel size of `(3,3)` which detects low-level features. And 2 convolutional layers with kernel size `(5, 5)` attempt to detect gloabal features. The last convolutional layer has `(2, 2)` stride to reduce parameters of fully connected largely. This 5 convolutional layer perform better than 2-convolutional-layer LeNet network. 

Dropout layer is used after each fully connected layer to prevent overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_images]

Resized test images are:

![alt_text][resize_test_images]

And training samples images are:

![alt_text][train_images]

From images above, we can find a big difference between test images and training images. One difference is taht, traffic signs in training images are centered in images while test images are skewed. They are not homogeneous. In other words, they do not have the same distribution. So the classifier might not work well on test images.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			              |     Prediction	        					   					   					  | 
|:---------------------:|:---------------------------------------------:| 
| Road work      		     | Road work  					   					   					   					  				| 
| No entry     			      | No entry								   					   					   					   			|
| Do not enter					     | No entry								   					   					   					   			|
| Speed limit (30km/h)  | Beware of ice/snow				   					   					    				|
| Mandatory direction of travel		| Ahead only         								|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. Compared to model's performance on test set, it does not perform well.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Here are top 5 probability for new 5 test images.

For 1st image (Road work):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.62         			| Road work |
| 0.15 | Keep right |
| 0.12 | Keep left |
| 0.06 | Bicycles crossing |
| 0.05 | Children crossing |

For 2nd image (No entry):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.73         			| No entry |
| 0.12 | Keep right | 
| 0.06 | Wild animals crossing | 
| 0.06 | Bicycles crossing | 
| 0.02 | Traffic signals |

For 3rd image (Do not enter):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.67         			| No entry|
| 0.21| Speed limit (20km/h)|
| 0.16| Bicycles crossing|
| 0.04| Bumpy road|
| 0.02| Stop|

For 4th image (Speed limit (30km/h)):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.59 |Beware of ice/snow|
| 0.31		| Stop|
| 0.14 |Road work|
| 0.05 |Road narrows on the right|
| 0.01 |Ahead only|

For 5th image (Mandatory direction of travel):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.70         			| Ahead only|
| 0.14 |Go straight or left|
| 0.07 |Roundabout mandatory|
| 0.07 |End of no passing|
| 0.05 |Go straight or right|
