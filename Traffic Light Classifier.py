# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:27:13 2020

@author: HUY QUYEN NGO (Jason)
Project: Trafic Light Classifier using Convolutional Neural Network
"""

import cv2 # computer vision library
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')



# Image data directories (put the directories below)
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"



## Display an image in IMAGE_LIST (try finding a yellow traffic light!)
## Print out 1. The shape of the image and 2. The image's label
image_number = 0
selected_image = IMAGE_LIST[image_number][0]
label_selected_image = IMAGE_LIST[image_number][1]
print("Image shape: ", selected_image.shape)
print("Image label: ", label_selected_image)
plt.imshow(selected_image)



# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    ## Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32,32))
    return standard_im



## TOne hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label
# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]
def one_hot_encode(label):
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [] 
    light = 1
    if label == "red":
        one_hot_encoded = [light,0,0]
    if label == "yellow":
        one_hot_encoded = [0,light,0]
    if label == "green":
        one_hot_encoded = [0,0,light]
    return one_hot_encoded



'''
# Importing the tests
import test_functions
tests = test_functions.Tests()
# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)
'''



def standardize(image_list):
    # Empty image data array
    standard_list = []
    
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        
        # Standardize the image
        standardized_im = standardize_input(image)
        
        # One-hot encode the label
        one_hot_label = one_hot_encode(label)  
        
        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)



## Display a standardized image and its label
stan_num = 0
stan_im = STANDARDIZED_LIST[stan_num][0]
stan_im_label = STANDARDIZED_LIST[stan_num][1]
plt.imshow(stan_im)



# Convert and image to HSV colorspace
# Visualize the individual color channels
image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')



## Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def feature1(rgb_image):

    ## Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    ## Create and return a feature value and/or vector
    
    # Crop the image to desired size
    image_crop = np.copy(hsv)
    image_crop = hsv[3:-3, 11:-11, :]
    
    ## Define the brightness value boundaries in HSV values
    hsvlower = np.array([0,0,0])
    hsvupper = np.array([255,255,200])

    ## Define the masked area and mask the image
    mask = cv2.inRange(image_crop, hsvlower, hsvupper)

    # Mask the image to get the light only
    masked_image = np.copy(image_crop)
    masked_image[mask != 0] = [0, 0, 0]
    
    #plt.imshow(masked_image)
    
    # Average brightness values of 3 regions top, middle and bottom
    part_area = 8*10
    avg_bright_red = (np.sum(image_crop[0:9,:,2]))/part_area
    avg_bright_yellow = (np.sum(image_crop[9:17,:,2]))/part_area
    avg_bright_green = (np.sum(image_crop[17:25,:,2]))/part_area
    #print("Average bright of red light: ", avg_bright_red)
    #print("Average bright of yellow light: ", avg_bright_yellow)
    #print("Average bright of green light: ", avg_bright_green)
    
    brightness = [avg_bright_red,avg_bright_yellow,avg_bright_green]
    
    return brightness



# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    # Initialize the predicted label to be green light
    predicted_label = [0,0,1]
    
    # If 1st region has the highest average brightness, the light is red
    # If 2nd region has the highest average brightness, the light is yellow
    brightness = feature1(rgb_image)
    if max(brightness) == brightness[0]:
        predicted_label = [1,0,0]
    if max(brightness) == brightness[1]:
        predicted_label = [0,1,0]
    
    return predicted_label   



# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)



# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))



# Visualize misclassified example(s)
## Display an image in the `MISCLASSIFIED` list 
## Print out its predicted label - to see what the image *was* incorrectly classified
misclassified_num = 0
misclassified_im = MISCLASSIFIED[misclassified_num][0]
misclassified_im_label = MISCLASSIFIED[misclassified_num][1]
print("Misclassified label: ", misclassified_im_label)
plt.imshow(misclassified_im)



'''
# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
'''