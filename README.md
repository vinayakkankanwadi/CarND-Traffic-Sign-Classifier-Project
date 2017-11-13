## Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="writeup-images/traffic-signs-classifier.PNG" width="1000" alt="Combined Image" />

Overview
---
In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. Specifically, we will train a model to classify traffic signs from the German Traffic Sign Dataset. The trained model is then tested on German traffic signs found on the web.

Goals
---
* Step 1: Load the data set (see below for links to the project data set)
* Step 2: Explore, summarize and visualize the data set
* Step 3: Design, train and test a model architecture
* Step 4: Use the model to make predictions on new images
* Step 5: Analyze the softmax probabilities of the new images
* Step 6: Summarize the results with a written report

Files
---
* [Traffic_Sign_Classifier.ipynb](https://github.com/vinayakkankanwadi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) - Notebook file with all questions answered and all code cells executed and displaying output 
* [Report.html](https://github.com/vinayakkankanwadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) - An HTML export of the project notebook.
* [Test-Images](https://github.com/vinayakkankanwadi/CarND-Traffic-Sign-Classifier-Project/tree/master/test-images)- Images used for the project that are not from the German Traffic Sign Dataset.
* [German Traffic Sign Dataset ZIP](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) - This is not included NEED TO BE DOWNLOADED, UNZIPPED and copy train.p, test.p and valid.p to [Traffic-signs-data](https://github.com/vinayakkankanwadi/CarND-Traffic-Sign-Classifier-Project/tree/master/traffic-signs-data) folder.
* [README.md](https://github.com/vinayakkankanwadi/CarND-Traffic-Sign-Classifier-Project/blob/master/README.md) - Writeup report as a markdown to reflect the work.


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


### Step 1: Data Set Summary & Exploration

#### 1a. Dataset Summary

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 1b. Exploratory Visualization

The bar chart shows the data distribution of the training, testing and validation data. Each bar represents one class (traffic sign) and how many samples are in their respective class distribution. The mapping of traffic sign names to class id can be found here: [signnames.csv](./signnames.csv)

![histogram](./writeup-images/Exploratory%20Visualization.png "histogram")

Sample traffic signs from the training data set. More can be found in the [jupyter notebook](./Traffic_Sign_Classifier.ipynb).

![Sample training images](./writeup-images/exploratory-visualization-sample.PNG "sample training images")


### Step 2: Design and Test a Model Architecture

#### 2a. Preprocessing
Steps
- Convert to Grayscale - Color unlikely to give any performance boost.
- Normalization - to reduce the number of shades.
- AUGMENT THE TRAINING DATA - generate additional dataset - common technique to improve model's precision. Rotate(15 degree) and Scale(1.25 percentage) used.

Here is an example of an original image and pre-processed image:

![Pre-process training images](./writeup-images/preprocessing.PNG "Pre-process training images")


#### 2b. Model Architecture

Convolutional neuronal network Model is used to classify the traffic signs. The input of the network is an 32x32x1 image and the output is the probabilty of each of the 43 possible traffic signs.
 
 Model consisted of the following layers:

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x6|
| Max pooling			| 2x2 stride		|28x28x6|14x14x6|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x6|10x10x16|
| Max pooling			| 2x2 stride	   					|10x10x16|5x5x16|
| Flatten				| 3 dimensions -> 1 dimension					|5x5x16| 400|
| Fully Connected | connect, RELU, Dropout (keep prob = 0.75)			|400|120|
| Fully Connected | connect, RELU, Dropout (keep prob = 0.75)   |120|84|
| Fully Connected | connect, RELU, Dropout (keep prob = 0.75)  	|84|**43**|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


