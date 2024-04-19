from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os

BASE_OUTPUT = "Output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
print(MODEL_PATH)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image paths")
args = vars(ap.parse_args())

# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:    
	# load the image paths in our testing file       
	imagePaths = open(args["input"]).read().strip().split("\n")

# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
model = load_model(MODEL_PATH)

model.summary()
lb = pickle.loads(open(LB_PATH, "rb").read())

indexer = 1
for imagePath in imagePaths:
	"""
	Loading the input image (in Keras format) from disk and preprocess
	it, scaling the pixel intensities to the range [0, 1]
	"""
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)
	# Predict the bounding box of the object along with the class label
	(boxPreds, labelPreds) = model.predict(image)
	print("BoxPreds\n", boxPreds)
	print("\nLabelPreds\n", labelPreds)
	(startX, startY, endX, endY) = boxPreds[0]
	# Determine the class label with the largest predicted probability
	i = np.argmax(labelPreds, axis=1)
	label = lb.classes_[i][0]
	print(lb.classes_[i][0])
	"""
	Loading the input image (in OpenCV format), resize it such that it
	fits on our screen, and grab its dimensions
	"""
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	# Scaling the predicted bounding box coordinates based on the image dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)
	print(w, h, startX, startY, endX, endY)
	# Drawing the predicted bounding box and class label on the image
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, str(label), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
	name = "Test_" + str(indexer) + ".jpg"
	path = os.path.sep.join([BASE_OUTPUT, name])
	cv2.imwrite(path, image)
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	indexer += 1
	#exit()

