import tensorflow as tf

def fgsm_attack(model, input_image, input_label,  epsilon):
  """
  Implementing the FGSM attack to create a destructive instance
  model: Deep learning model
  input_image: Input image (TensorFlow tensor)
  input_label: Correct label (TensorFlow tensor)
  epsilon: Attack intensity (ε)
  """
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = tf.keras.losses.CategoricalCrossentropy()(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  perturbations = signed_grad
  # add attack to the pic
  adv_x = image + epsilon*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  return adv_x, perturbations


def pgd_attack(model, input_image, input_label, epsilon, alpha, num_iterations):
  """
  Implementing the PGD attack to create a destructive instance
  model: Deep learning model
  input_image: Input image (TensorFlow tensor)
  input_label: Correct label (TensorFlow tensor)
  epsilon: Attack intensity (ε)
  alpha: Step size per iteration (α)
  num_iterations: Number of iterations (T).
  """
  # Destructive image initialization
  adversarial_image = input_image + tf.random.uniform(input_image.shape, -epsilon, epsilon)
  adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)

  for _ in range(num_iterations):
      with tf.GradientTape() as tape:
          tape.watch(adversarial_image)
          prediction = model(adversarial_image)
          loss = tf.keras.losses.CategoricalCrossentropy()(input_label, prediction)

      # Calculating gradients
      gradient = tape.gradient(loss, adversarial_image)
      # Calculating the sign of gradients
      signed_grad = tf.sign(gradient)
      perturbations = signed_grad
      # Destructive image update
      adversarial_image = adversarial_image + alpha * signed_grad
      # Projecting within the permitted limits
      adversarial_image = tf.clip_by_value(adversarial_image, input_image - epsilon, input_image + epsilon)
      adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)

  return adversarial_image, perturbations


def bim_attack(model, input_image, input_label, epsilon, alpha, num_iterations):
  """
  Implementing the BIM attack to create a destructive instance
  model: Deep learning model
  input_image: Input image (TensorFlow tensor)
  input_label: Correct label (TensorFlow tensor)
  epsilon: Attack intensity (ε)
  alpha: Step size per iteration (α)
  num_iterations: Number of iterations (T).
  """
  # Destructive image initialization
  adversarial_image = input_image

  for _ in range(num_iterations):
      with tf.GradientTape() as tape:
          tape.watch(adversarial_image)
          prediction = model(adversarial_image)
          loss = tf.keras.losses.CategoricalCrossentropy()(input_label, prediction)

      # Calculating gradients
      gradient = tape.gradient(loss, adversarial_image)
      # Calculating the sign of gradients
      signed_grad = tf.sign(gradient)
      perturbations = signed_grad
      # Destructive image update
      adversarial_image = adversarial_image + alpha * signed_grad
      # Projecting within the permitted limits
      adversarial_image = tf.clip_by_value(adversarial_image, input_image - epsilon, input_image + epsilon)
      adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)

  return adversarial_image, perturbations


def mi_fgsm_attack(model, input_image, input_label, epsilon, alpha, num_iterations, momentum):
  """
  Implementing the MI_FGSM attack to create a destructive instance
  model: Deep learning model
  input_image: Input image (TensorFlow tensor)
  input_label: Correct label (TensorFlow tensor)
  epsilon: Attack intensity (ε)
  alpha: Step size per iteration (α)
  num_iterations: Number of iterations (T)
  momentum: Momentum parameter (μ)
  """

  # Destructive image initialization and cumulative gradient
  adversarial_image = input_image
  g = tf.zeros_like(input_image)

  for _ in range(num_iterations):
      with tf.GradientTape() as tape:
          tape.watch(adversarial_image)
          prediction = model(adversarial_image)
          loss = tf.keras.losses.CategoricalCrossentropy()(input_label, prediction)

      # Calculating gradients
      gradient = tape.gradient(loss, adversarial_image)
      # Normalizing gradients with L1 normal
      gradient_norm = tf.norm(gradient, ord=1)
      normalized_gradient = gradient / gradient_norm
      # Cumulative Gradient Update with Momentum
      g = momentum * g + normalized_gradient
      # Calculating the sign of the cumulative gradient
      signed_grad = tf.sign(g)
      perturbations = signed_grad
      # Destructive image update
      adversarial_image = adversarial_image + alpha * signed_grad
      # Projecting within the permitted limits
      adversarial_image = tf.clip_by_value(adversarial_image, input_image - epsilon, input_image + epsilon)
      adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)

  return adversarial_image, perturbations
  
  
