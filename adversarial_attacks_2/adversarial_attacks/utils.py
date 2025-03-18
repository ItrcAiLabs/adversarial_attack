def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image
  
# Function to check model predictions
def predict(model, image):
    """
  Model prediction for input image.
  model: Deep learning model (TensorFlow/Keras)
  image: Input image (TensorFlow tensor)
    """
    prediction = model(image)
    return tf.argmax(prediction, axis=1).numpy()[0]