# adversarial_attacks/attacks.py
import tensorflow as tf
from .utils import predict
import numpy as np
import matplotlib.pyplot as plt


#####Data poisoning(Gradient Matching / Witches' Brew Attack)
def create_poisoned_data(dataset, target_gradient, poison_rate=0.1):
    """
  Implementing the Gradient Matching / Witches' Brew Attack

   dataset: The original dataset.
   target_gradient: The target gradient.
   poison_rate: The poisoning rate.
  return: The poisoned dataset.
    """
    poisoned_images = []
    poisoned_labels = []
    for image, label in dataset:
        if np.random.rand() < poison_rate:
            # Image gradient calculation
            with tf.GradientTape() as tape:
                tape.watch(image)
                # Let's assume we have a simple model and calculate the gradient relative to it.
                # In a real implementation, the gradient must be calculated relative to the target model.
                prediction = tf.reduce_sum(image)
            image_gradient = tape.gradient(prediction, image)

            image = image + (target_gradient - image_gradient) * 0.1

            # Change label to target label
            label = 0
        poisoned_images.append(image)
        poisoned_labels.append(label)
    return tf.data.Dataset.from_tensor_slices((poisoned_images, poisoned_labels))
#######################################
##########Query attacks##############
#Function to implement zoo_attack
def zoo_attack(model, input_image, target_label, epsilon=0.01, h=0.001, max_iter=100):
    """
  ZOO attack implementation.
  model: Deep learning model (TensorFlow/Keras)
  input_image: Input image (TensorFlow tensor)
  target_label: Target label (integer)
  epsilon: Attack intensity (ε)
  h: Step size for calculating approximate gradients
  max_iter: Maximum number of iterations
    """
    # Convert target label to one-hot format
    target_label = tf.one_hot(target_label, depth=1000)  # 1000 classes for ImageNet
    target_label = tf.reshape(target_label, (1, 1000))

    # Initialize the adversary image with the input image.
    adversarial_image = tf.identity(input_image)

    for i in range(max_iter):
        # Creating a random vector to calculate approximate gradients
        perturbation = tf.random.normal(adversarial_image.shape, mean=0, stddev=1)
        perturbation = perturbation / tf.norm(perturbation)

        # Calculating the cost function at two points (f(x + h*v) and f(x - h*v))
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            prediction = model(adversarial_image + h * perturbation)
            loss_plus = tf.keras.losses.categorical_crossentropy(target_label, prediction)

        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            prediction = model(adversarial_image - h * perturbation)
            loss_minus = tf.keras.losses.categorical_crossentropy(target_label, prediction)

        # Approximate gradient calculation
        gradient = (loss_plus - loss_minus) / (2 * h) * perturbation

        # Hostile image update
        adversarial_image = adversarial_image - epsilon * gradient
        adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)  

        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss_plus.numpy()}")

    return adversarial_image

# Function to implement Boundary Attack
def boundary_attack(model, input_image, target_label, max_iter=1000, epsilon=0.01):
    """
  Implement Boundary Attack.
  model: Deep learning model (TensorFlow/Keras)
  input_image: Input image (TensorFlow tensor)
  target_label: Target label (integer)
  max_iter: Maximum number of iterations
  epsilon: Attack intensity (ε)
    """
    # Initialize the adversary image with the input image.
    adversarial_image = tf.identity(input_image)

    for i in range(max_iter):
        # Creating a random vector to move along the decision boundary
        perturbation = tf.random.normal(adversarial_image.shape, mean=0, stddev=1)
        perturbation = perturbation / tf.norm(perturbation)

        # Hostile image update
        adversarial_image = adversarial_image + epsilon * perturbation
        adversarial_image = tf.clip_by_value(adversarial_image, -1, 1) 

        predicted_label = predict(model, adversarial_image)
        if predicted_label == target_label:
            print(f"Iteration {i}: Success! Predicted label = {predicted_label}")
            break

        if i % 100 == 0:
            print(f"Iteration {i}: Predicted label = {predicted_label}")

    return adversarial_image 
  
# HopSkipJumpAttack  
# Function to implement HopSkipJumpAttack
def hopskipjump_attack(model, input_image, target_label, max_iter=100, epsilon=0.01, delta=0.01):
    """
  Implementation of HopSkipJumpAttack attack.
  model: Deep learning model (TensorFlow/Keras)
  input_image: Input image (TensorFlow tensor)
  target_label: Target label (integer)
  max_iter: Maximum number of iterations
  epsilon: Attack severity (ε)
  delta: Step size for updating
    """
    # Initialize the adversary image with the input image.
    adversarial_image = tf.identity(input_image)

    # Main attack ring
    for i in range(max_iter):
        # Creating a random vector for hopping
        perturbation = tf.random.normal(adversarial_image.shape, mean=0, stddev=1)
        perturbation = perturbation / tf.norm(perturbation)

        # Hostile image update with hop
        adversarial_image = adversarial_image + epsilon * perturbation
        adversarial_image = tf.clip_by_value(adversarial_image, -1, 1) 

        # Checking the model's predictions
        predicted_label = predict(model, adversarial_image)
        if predicted_label == target_label:
            print(f"Iteration {i}: Success! Predicted label = {predicted_label}")
            break

        # Update the hostile image with Skip
        gradient = tf.random.normal(adversarial_image.shape, mean=0, stddev=1)
        gradient = gradient / tf.norm(gradient)
        adversarial_image = adversarial_image + delta * gradient
        adversarial_image = tf.clip_by_value(adversarial_image, -1, 1) 

        predicted_label = predict(model, adversarial_image)
        if predicted_label == target_label:
            print(f"Iteration {i}: Success! Predicted label = {predicted_label}")
            break

        if i % 10 == 0:
            print(f"Iteration {i}: Predicted label = {predicted_label}")

    return adversarial_image
 
#####################Digital attacks###############
 # Function to create universal_perturbation
def universal_perturbation(model, images, max_iter=50, epsilon=0.1, delta=0.2):
    """
  Create global perturbation.
  model: Deep learning model (TensorFlow/Keras)
  images: List of input images (TensorFlow tensor)
  max_iter: Maximum number of iterations
  epsilon: Perturbation intensity (ε)
  delta: Threshold to stop
    """
    # 
    perturbation = tf.zeros_like(images[0])

    # Initializing the disturbance
    for i in range(max_iter):
        for image in images:
            # Add noise to the image
            perturbed_image = image + perturbation
            perturbed_image = tf.clip_by_value(perturbed_image, -1, 1)  

            # Checking the model's predictions
            original_label = predict(model, image)
            perturbed_label = predict(model, perturbed_image)

            # If the model is wrong, update the perturbation.
            if original_label != perturbed_label:
                continue

            with tf.GradientTape() as tape:
                tape.watch(perturbed_image)
                prediction = model(perturbed_image)
                loss = -tf.reduce_max(prediction) 

            gradient = tape.gradient(loss, perturbed_image)
            perturbation = perturbation + epsilon * tf.sign(gradient)
            perturbation = tf.clip_by_value(perturbation, -delta, delta)  

        if i % 10 == 0:
            print(f"Iteration {i}: Perturbation norm = {tf.norm(perturbation).numpy()}")

    return perturbation 


# Function to implement DeepFool attack
def deepfool_attack(model, input_image, max_iter=50, overshoot=0.02):
    """
  Implementation of the DeepFool attack.
  model: Deep learning model (TensorFlow/Keras)
  input_image: Input image (TensorFlow tensor)
  max_iter: Maximum number of iterations
  overshoot: Parameter to prevent overestimation
    """
    # Initialize the adversary image with the input image.
    adversarial_image = tf.identity(input_image)
    adversarial_image = tf.Variable(adversarial_image) 

    original_label = predict(model, adversarial_image)

    for i in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            predictions = model(adversarial_image)
            original_prediction = predictions[0, original_label]

        gradients = tape.gradient(original_prediction, adversarial_image)

        perturbation = tf.sign(gradients)
        adversarial_image.assign_add(perturbation * overshoot)

        predicted_label = predict(model, adversarial_image)
        if predicted_label != original_label:
            print(f"Iteration {i}: Success! Predicted label = {predicted_label}")
            break

        if i % 10 == 0:
            print(f"Iteration {i}: Predicted label = {predicted_label}")

    return adversarial_image  