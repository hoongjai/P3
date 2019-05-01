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

[image1]: ./output_images/visual_train_data.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/12Priority_resized.jpg "Traffic Sign 1"
[image5]: ./test_images/14Stop_sign_resized.jpg "Traffic Sign 2"
[image6]: ./test_images/25working_resized.jpg "Traffic Sign 3"
[image7]: ./test_images/31animal_resized.jpg "Traffic Sign 4"
[image8]: ./test_images/34turn_left_resized.jpg "Traffic Sign 5"
[top_five_soft_max1]: ./output_images/top_five_softmax1.jpg "1"
[top_five_soft_max2]: ./output_images/top_five_softmax2.jpg "2"
[top_five_soft_max3]: ./output_images/top_five_softmax3.jpg "3"
[top_five_soft_max4]: ./output_images/top_five_softmax4.jpg "4"
[top_five_soft_max5]: ./output_images/top_five_softmax5.jpg "5"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I printed out 5 images from each category to checked how they look. Apparently they looked similar therefore I took the 1st, 40th, 80th, 120th, 160th pictures for visual check.

The occurrences of each category is printed as title. For instance, Speed Limit (20km/s) had 180 pictures while Road Work had 1350 pictures.

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I didn't convert the image to grayscale and it gave me bad accuracy. Therefore I decided to convert the images to grayscale.

Next, I normalized the image data since this will help improve the performance. 

At first I tried with the original LeNet from CNN class but the accuracy wasn't meeting the requirement. Due to overfitting the accuracy drop. 
I improved the accuracy by adding dropout of 0.5 and following parameters:
EPOCHS = 50
BATCH_SIZE = 128
learing rate = 0.0009


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6 									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten | output 400
| Fully connected		| output 120      									|
| RELU					|												|
| Dropout ||
| Fully connected		| output 84      									|
| RELU					|												|
| Dropout ||
| Fully connected		| output 43      									|
| Softmax				| etc.        									|
 


To train the model, I used an adam optimizer and following parameters:
Epochs: 50
Batch size: 128
Learning rate: 0.0009
Dropout: 0.5
mu: 0
Sigma = 0.1


My final model results were:
* training set accuracy of 95.7%
* validation set accuracy of 95.7%
* test set accuracy of 93.7%

I used original LeNet model as first choice and the result is below required 93%. To used original LeNet copied from class room, I change the image size from 32x32x3 to 32x32x1 and the label number from 10 to 43 (traffice signs).
Epochs: 10, batch size: 128 and learning rate of 0.001 were used.

Following are problems observed from original LeNet model:
Overfitting is observed.
Learning rate is too high.
Training cycles, EPochs is too low.

The adjustments I made to the model:
After trial and error I added 2 dropouts after RELU activation functions.
Set the learining rate to 0.0009
Epochs increased to 50.

The final model accuracy is 80% where 4 out of 5 test samples were detected correctly.
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%! 

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

The top five soft max probabilities were
![top_five_soft_max1]
![top_five_soft_max2]
![top_five_soft_max3]
![top_five_soft_max4]
![top_five_soft_max5]

