import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os
import csv
import pickle

from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

from tensorflow.keras.applications import DenseNet201 # Output
from tensorflow.keras.applications import NASNetMobile # Output-2
from tensorflow.keras.applications import Xception # Output-3

BASE_PATH = "../Boreal/Ruokolahti_annotations"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "imgs"])
ANNOS_PATH = os.path.sep.join([BASE_PATH, "annotations/csv"])

BASE_OUTPUT = "Output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])

INIT_LR = 1e-4
NUM_EPOCHS = 300
BATCH_SIZE = 32

print("[INFO] loading dataset...")

data = []
labels = []
annos = []
image_paths = []

for file_name in os.listdir(ANNOS_PATH):
    file_path = os.path.join(ANNOS_PATH, file_name)
    with open(file_path, 'r') as file:
         
         reader_object = csv.reader(file)
         # Skip first line, that holds column names
         next(reader_object)
         
         for row in reader_object:
             
             if row[0] == "Big smoke":
                 #--------------------------
                 print(row[0], ", ", row[1], ", ", row[2], ", ", row[3], ", ", row[4])
                 label_name = row[0]
                 #print(row[0])
                 start_x = float(row[1])
                 start_y = float(row[2])
                 bbox_width = float(row[3])
                 bbox_height = float(row[4])
                 image_name = row[5]
                 w = float(row[6])
                 h = float(row[7])
                 img_path = os.path.sep.join([IMAGES_PATH, image_name])
                 #--------------------------

                 #---- Anno -------------------------------
                 # Yolo bbox to TF bbox anno-format
                 
                 """
                 startX = float(center_x - bbox_width / w)
                 startY = float(center_y - bbox_height / h)
                 endX = float(center_x + bbox_width / w)
                 endY = float(center_y + bbox_height / h)
                 """
                 end_x = float(start_x + bbox_width)
                 end_y = float(start_y + bbox_height)
                 
                 # Spatial scaling
                 
                 startX = float(start_x) / w
                 startY = float(start_y) / h
                 endX = float(end_x) / w
                 endY = float(end_y) / h
                 
                 #-----------------------------------------
                 
                 image = load_img(img_path, target_size=(224, 224))
                 image = img_to_array(image)

                 data.append(image)
                 annos.append((startX, startY, endX, endY))
                 labels.append(label_name)
                 image_paths.append(img_path)
                 
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
annos = np.array(annos, dtype="float32")
image_paths = np.array(image_paths)

#labels = np.ones(len(labels))

# For multiclass:

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) == 2:
    labels = to_categorical(labels)


split = train_test_split(data, labels, annos, image_paths,
                         test_size=0.20, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

output_dir = os.path.dirname(TEST_PATHS)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(TEST_PATHS, "w") as f:
    f.write("\n".join(testPaths))

f = open(TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

#architecture = DenseNet201(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
#architecture = NASNetMobile(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
#architecture = Xception(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
architecture = MobileNetV3Small(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

architecture.trainable = False
flatten = architecture.output
#flatten = Flatten(data_format='channels_last')(flatten)
flatten = Flatten()(flatten)

bounding_box_head = Dense(128, activation="relu")(flatten)
bounding_box_head = Dense(64, activation="relu")(bounding_box_head)
bounding_box_head = Dense(32, activation="relu")(bounding_box_head)
bounding_box_head = Dense(4, activation="sigmoid", name="bounding_box")(bounding_box_head)

softmax_head = Dense(512, activation="relu")(flatten)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(512, activation="relu")(softmax_head)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmax_head)

model = Model(inputs=architecture.input, outputs=(bounding_box_head, softmax_head))

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}
"""
losses = {
    "class_label": "binary_crossentropy",
    "bounding_box": "mean_squared_error",
}
"""

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

opt = Adam(learning_rate=INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics={"class_label": "accuracy", "bounding_box": "accuracy"}, loss_weights=lossWeights)
print(model.summary())

trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}

testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1,
    callbacks=[early_stopping_monitor]    
)

print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
print("[INFO] saving label binarizer...")
with open(LB_PATH, "wb") as f:
    pickle.dump(lb, f)
