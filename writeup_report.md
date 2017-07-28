# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[nvidia]: ./images/nvidia.PNG "NVIDIA ConvNet Architecture"
[out-of-track]: ./images/out-of-track.png "car out of track"
[center-lane-driving]: ./images/center-lane-driving.png "center lane driving"
[left-cam]: ./images/left-cam.jpg "left cam"
[center-cam]: ./images/center-cam.jpg "center cam"
[right-cam]: ./images/right-cam.jpg "right cam"
[mean-squared-error-loss]: ./images/mean-squared-error-loss.png "model mean squared error loss"

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 5x5 convolution layers and 1 3x3 convolution layer with depths between 24 and 64 (model.py lines 74-77) and 4 fully connected layers (model.py lines 79-83)

The model includes RELU layers after each convolution layer to introduce nonlinearity (model.py lines 74-77), and the data is normalized in the model using a Keras lambda layer (code line 68). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 19). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Dropout layers have not been needed.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy was to start with a simple neural network to show that the training pipeline is working and the simulator can use the model to steer the car. 

Having proven that the pipeline was working, I used a known network architecture (LeNet) as a starting point. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

I run the simulator to see how well the car was driving around the track. There were a few spots where the vehicle fell off the track.

With a more appropriate architecture (NVIDIA) and an augmented data set, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Finally, I used a model architecture like NVIDIA in their paper 'End-To-End Deep Learning for Self-Driving Cars'  (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

![alt text][nvidia]

The final model architecture (model.py lines 74-82) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         | Size |    Details      |
| ------------- |-----:| ---------------------------------|
| Normalization |  3  | input shape 160x320x3            |
| Cropping2D    |  3   | top crop=70, bottom crop=25      |
| Convolution2D | 24  | 5x5, stride 2x2, activation=relu |
| Convolution2D | 36  | 5x5, stride 2x2, activation=relu|
| Convolution2D | 48  | 5x5, stride 2x2, activation=relu |
| Convolution2D | 64  | 3x3, stride 2x2, activation=relu |
| Flatten       |     |                                  |
| Dense         | 100 |                                  |
| Dense         | 50  |                                  |
| Dense         | 10  |                                  |
| Dense         | 1   |                                   

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap using center lane driving. Here is an example image of center lane driving:

![alt text][center-lane-driving]

The captured data of the simulater comprised 3 images per time step from 3 different cameras: left, center and right camera.

To teach the car to recover from the left and right sides of the road, I used the images from the left and right cameras and adjusted the steering angles for the left and right camera images.

left cam

![alt text][left-cam] 

center cam

![alt text][center-cam] 

right cam

![alt text][right-cam]

| camera | angle adjustment value|
|--------|------------|
| left   | +0.2       |
| center | 0          |
| right  | +0.2      |

To augment the data set, I also flipped images and angles. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. To have short training times, I only used 2 epochs. 

![alt text][mean-squared-error-loss]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

I found out that one lap of data was not enough to train the model so that it would be able to keep the car on the road. Finally, I used a data set containing 4 laps with 24.498 images.

With this big data set, my python script ran out of memory. To use less memory, I used a python generator (model.py lines 21-62) to read the images in batches during training (model.py line 86).

After training the model with the data of 4 laps, there was still a critical location on the track where the car went of the road. 

![alt text][out-of-track]

After increasing the adjustment value of the left and right camera image to +-0.4, the car managed to stay on the road.

Here is a link to the [video](../master/video.mp4) showing the car driving one lap without touching the borders of the road.
