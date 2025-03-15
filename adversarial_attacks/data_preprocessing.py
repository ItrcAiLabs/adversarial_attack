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

def extract_ids(image_folder:str, ann_file:str):
  """
    return final_ids which is in pic and ann file
    image_folder: path to the image folder
    ann_file: path to the annotation file

  """
  coco = COCO(ann_file)
  # List of ids in ann
  img_ids = coco.getImgIds()

  # path of images
  image_list = os.listdir(image_folder)
  id_list = []
  for image_name in image_list:
    if image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
      # Extracting ID using regular expression
      match = re.search(r'\d+', image_name)
      if match:
        image_id = int(match.group())
        id_list.append(image_id)

  # Intersection of two list
  final_id = list(set(img_ids) & set(id_list))

  return final_id

def one_label_COCO(image_folder:str, ann_file:str, main_labels:dict):
  """
    return image id and uniq label (Select the class with the largest number of objects)
    which filtered by main_label and return sorted list of keys
    image_folder: path to the image folder
    ann_file: path to the annotation file
    main_labels: main labels we need (must be in {})
  """
  # retirn final_ids which is in pic and ann file
  image_ids = extract_ids(image_folder, ann_file)
  # Define coco
  coco = COCO(ann_file)

  # Create a dictionary to store the final labels
  img_to_label = {}

  # Processing each image
  for img_id in image_ids:
      # Get image annotations
      ann_ids = coco.getAnnIds(imgIds=img_id)
      anns = coco.loadAnns(ann_ids)

      # Count the number of objects of each class
      class_counts = {}
      for ann in anns:
          class_id = ann['category_id']
          class_counts[class_id] = class_counts.get(class_id, 0) + 1

      # Select the class with the largest number of objects.
      if class_counts:
          dominant_class = max(class_counts, key=class_counts.get)
          img_to_label[img_id] = dominant_class
      else:
          img_to_label[img_id] = None  # If the image has no objects

  ## Print number of label
  #lebel count
  lebel_counts = Counter(img_to_label.values())
  # Convert to list of tuples and sort by count (highest to lowest)
  sorted_value_counts = sorted(lebel_counts.items(), key=lambda x: x[1], reverse=True)
  #print them
  for value, count in sorted_value_counts:
      print(f"Value = {value}: {count} times")

  #filtered by labels
  filtered_dict = {k: v for k, v in img_to_label.items() if v in main_labels}
  #Sort Dict by id
  filtered_dict = dict(sorted(filtered_dict.items()))
  #Sorted Key list
  sorted_keys_list = sorted(filtered_dict.keys())

  return filtered_dict, sorted_keys_list

def onehot_labels(my_dict:dict, num_classes:int):
  """
  changes the dictionary which has id and classes to one hot array
  my_dict: main dictionary
  num_classes: number of classes
  """
  # Number of classes
  num_classes = num_classes
  # Create a 2D array of the form (number of data, num_classes)
  num_data = len(my_dict)
  onehot_array = np.zeros((num_data, num_classes))
  # Converting a dictionary to a one-hot two-dimensional array
  for idx, (key, val) in enumerate(my_dict.items()):
      onehot_array[idx][val] = 1
  return onehot_array


def pic_preproc_Intersection(image_folder:str, ann_file:str, main_labels:dict, size: int):
  """
  resize and add 1 dimention to the image.  return just Intersection images.
  image_folder: path to the image folder
  ann_file: path to the annotation file
  main_labels: main labels we need (must be in {})
  size: size of Length and width
  """
  _,id_Intersection = one_label_COCO(image_folder, ann_file, main_labels)
  # A list to save the paths of the desired images
  filtered_image_paths = []

  # Reading image paths
  image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(".jpg")]

  # Filter image paths based on IDs in id_Intersection
  for image_path in image_paths:
      image_name = os.path.basename(image_path)
      id_number = int(image_name.split('.')[0])  #Extracting ID from image name
      if id_number in id_Intersection:
          filtered_image_paths.append(image_path)  # Add image path to filtered list
  # Sort the path
  filtered_image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
  print(filtered_image_paths)
  #####Preprocess function
  all_images = []
  # main function
  for img_path in filtered_image_paths:
      img = cv2.imread(img_path)
      gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      resized_img = cv2.resize(gray_img1, (size, size))
      gray_img2 = resized_img[:, :, np.newaxis]
      # Normalize pixel values to be between 0 and 1
      gray_img2 = gray_img2 / 255.0

      # Add processed image to list
      all_images.append(gray_img2)
  # Convert list to NumPy array
  all_pic = np.array(all_images)

  return all_pic

def pic_preproc_Intersection_3chan(image_folder:str, ann_file:str,main_labels:dict, size: int):
  """
  resize the image.  return just Intersection images.
  image_folder: path to the image folder
  ann_file: path to the annotation file
  main_labels: main labels we need (must be in {})
  size: size of Length and width
  """
  _,id_Intersection = one_label_COCO(image_folder, ann_file, main_labels)
  # A list to save the paths of the desired images
  filtered_image_paths = []

  # Reading image paths
  image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(".jpg")]

  # Filter image paths based on IDs in id_Intersection
  for image_path in image_paths:
      image_name = os.path.basename(image_path)
      id_number = int(image_name.split('.')[0])  #Extracting ID from image name
      if id_number in id_Intersection:
          filtered_image_paths.append(image_path)  # Add image path to filtered list
  # Sort the path
  filtered_image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
  print(filtered_image_paths)
  #####Preprocess function
  all_images = []
  # main function
  for img_path in filtered_image_paths:
      img = cv2.imread(img_path)
      resized_img = cv2.resize(img, (size, size))
      # Normalize pixel values to be between 0 and 1
      resized_img = resized_img / 255.0
      # Add processed image to list
      all_images.append(resized_img)
  # Convert list to NumPy array
  all_pic = np.array(all_images)

  return all_pic

def train_test_splitt(x_all:np.ndarray, y_all:np.ndarray, test_ratio:float):
  """
  split train and test. for x and y
  x_all: x data (input pic)
  y_all: y data (label)
  test_ratio:  test ratio (float)
  """
  # Splitting the data into train and test with a ratio of 80 to 20 percent
  x_train, x_test, y_train, y_test,  = train_test_split(x_all,y_all, test_size=test_ratio , random_state=42)
  return x_train, x_test, y_train, y_test
