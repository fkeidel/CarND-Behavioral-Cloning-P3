import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Convolution2D, Activation, MaxPooling2D, Cropping2D, BatchNormalization
import matplotlib.pyplot as plt

def readDrivingLog(path):
    lines = []
    with open (path+'driving_log.csv') as csvfile:
        csvreader = csv.reader (csvfile)
        for line in csvreader:
            lines.append (line)
    return lines

def readImage(source_path, current_path):
    filename = source_path.split('\\')[-1]
    image = cv2.imread(current_path+filename)
    return image

def getData(lines, data_path):
    images = []
    measurements =[]

    for line in lines:
        steering_center = float (line[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        image_path = data_path + 'IMG/'
        img_center = readImage(line[0],image_path)
        img_left = readImage(line[1],image_path)
        img_right = readImage(line[2],image_path)

        # add images and angles to data set
        images.extend([img_center, img_left, img_right])
        measurements.extend([steering_center, steering_left, steering_right])
    return (images, measurements)

def augmentData(images, measurements):
    augmented_images = []
    augmented_measurements = []

    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement)
        augmented_measurements.append (measurement*-1.0)
    return(augmented_images, augmented_measurements)

def leNet():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(6, 5, 5,activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.50))
    model.add(Convolution2D(6, 5, 5,activation='relu'))
    #model.add(Convolution2D(16, 5, 5,activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia(): #https://discussions.udacity.com/t/resourceexhaustederror-with-oom/228476/3
    #ch, row, col = 3, 160, 320 # image format
    keep_prob = 0.5
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3),output_shape=(160,320,3)))
    #model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, ch),output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70,25),(1,1))))
    model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def nvidia1(): #https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(160,320,3)))
    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    return model

def plotLoss(history_object):
    ### print the keys contained in the history object
    print (history_object.history.keys ())
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    print('preprocessing')
    #data_path = 'data_Frank/merged/'
    data_path = 'data_Frank/forward/'
    print('read log')
    lines = readDrivingLog (data_path)
    print('read images')
    (images,measurements) = getData(lines, data_path)
    print('augment data')
    (augmented_images, augmented_measurements) = augmentData(images,measurements)
    X_train = np.array (augmented_images)
    y_train = np.array (augmented_measurements)
    print('create model')
    model = nvidia()
    print('compile')
    model.compile (loss='mse', optimizer='adam')
    print('train')
    history_object = model.fit (X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
    print('save')
    model.save ('model.h5')
    print('done')
    #plotLoss(history_object)