import os
import numpy
path, dirs, files = next(os.walk("./letters_mod3"))
files.sort()

from emnist import extract_training_samples

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

import cv2
import matplotlib.pyplot as plt
import matplotlib

import pickle


if (input("Do you want to train a model? yes/no ... ") == "yes" or ""):
    X, y = extract_training_samples('letters')

    # Make sure that every pixel in all of the images is a value between 0 and 1
    X = X / 255.

    # Use the first 60000 instances as training and the next 10000 as testing
    X_train, X_test = X[:60000], X[60000:70000]
    y_train, y_test = y[:60000], y[60000:70000]

    # There is one other thing we need to do, we need to
    # record the number of samples in each dataset and the number of pixels in each image
    X_train = X_train.reshape(60000,784)
    X_test = X_test.reshape(10000,784)

    print("Extracted our samples and divided our training and testing data sets")


    # img_index = 14000 # <<<<<  You can update this value to look at other images
    # img = X_train[img_index]
    # print("Image Label: " + str(chr(y_train[img_index]+96)))
    # plt.imshow(img.reshape((28,28)))

    # THE MAIN THING THAT WE WANT TO CONFIGURE ###

    mlp2 = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,), max_iter=100, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)

    # ##############################################

    mlp2.fit(X_train, y_train)
    print("Training set score: %f" % mlp2.score(X_train, y_train))
    print("Test set score: %f" % mlp2.score(X_test, y_test))

    # This code processes all the scanned images and adds them to the handwritten_story
    handwritten_story = []
    for i in range(len(files)):
        img = cv2.imread("./letters_mod3/"+files[i],cv2.IMREAD_GRAYSCALE)
        handwritten_story.append(img)

    print("Imported the scanned images.")

    plt.imshow(handwritten_story[4])  #<--- Change this index to see different letters
    with open("./models/trained.pickle", "wb") as f:
        pickle.dump(mlp2, f)
else: 
    pickle_in = open("models/trained.pickle", "rb")
    mlp2 = pickle.load(pickle_in)

typed_story = ""
for letter in handwritten_story:
  letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_CUBIC)
    
  #this bit of code checks to see if the image is just a blank space by looking at the color of all the pixels summed
  total_pixel_value = 0
  for j in range(28):
    for k in range(28):
      total_pixel_value += letter[j,k]
  if total_pixel_value < 20:
    typed_story = typed_story + " "
  else:         #if it NOT a blank, it actually runs the prediction algorithm on it
    single_item_array = (numpy.array(letter)).reshape(1,784)
    prediction = mlp2.predict(single_item_array)
    typed_story = typed_story + str(chr(prediction[0]+96))
    
print("Conversion to typed story complete!")
print(typed_story)