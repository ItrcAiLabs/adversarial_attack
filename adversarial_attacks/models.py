import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import re
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from google.colab.patches import cv2_imshow
from tensorflow.keras.applications import MobileNetV2



# CNN2D model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(183, activation='softmax'))


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Summary of the model
model.summary()


# Model training
history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))

# Model evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"test_acc: {test_acc:.4f}")