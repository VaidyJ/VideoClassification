# USAGE
# python train.py --dataset data --model model/activity.model --label-bin model/lb.pickle --epochs 80

# set the matplotlib backend so figures can be saved in the background
import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the set of labels from the spots activity dataset we are
# going to train our network on
LABELS = set(["forehand", "backhand"])

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# if the label of the current image is not part of of the labels
	# are interested in, then ignore the image
	if label not in LABELS:
		continue

	# load the image, convert it to RGB channel ordering, and resize
	# it to be a fixed 224x224 pixels, ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224)); image = image.astype(float)
    
    # update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.2, stratify=labels, random_state=101)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3))) # 3 channels for RGB

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel) # regularization to reduce overfitting
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
# using Stochastic gradient descent optiimizer
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
#optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
#model.compile(loss="binary_crossentropy", optimizer=opt,
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
#model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
	metrics
    =["accuracy"])



# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")

print("trainX ", len(trainX))
print("trainY ", len(trainY))

H = model.fit_generator(	trainAug.flow(trainX, trainY, shuffle=True, batch_size=32), 	steps_per_epoch=len(trainX) // 32, 	validation_data=valAug.flow(testX, testY, shuffle=True,), 	validation_steps=len(testX) // 32, epochs=args["epochs"])
# H = model.fit_generator(	trainAug.flow(data, labels, batch_size=32), 	steps_per_epoch=len(data) // 32, 	#validation_data=valAug.flow(testX, testY), 	validation_steps=len(testX) // 32, 
# epochs=args["epochs"])
	#epochs=args["epochs"])
    

# evaluate the network
print("[INFO] evaluating network...")
y_score = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	y_score.argmax(axis=1), target_names=lb.classes_))
#-------------------------Evaluation Curves---------------------------------------------------------



np.set_printoptions(precision=2)

y_test = testY.argmax(axis=1)
y_pred = y_score.argmax(axis=1)
lb = ["forehand", "backhand"] #Thunderstorm, Building_Collapse

#-----------------------------------------------------------------------
# serialize the model to disk
print("[INFO] Saving model...")
#model.save(args["model"])
model.save("model")

# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()