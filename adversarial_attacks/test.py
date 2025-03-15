import tensorflow as tf
from adversarial_attacks import preprocess, apply_attack, display_images

# Load your model
model = tf.keras.models.load_model("path/to/your/model")

# Load and preprocess the input image
image_path = "path/to/your/image.jpg"
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)
input_image = preprocess(image)

# Define the true label
true_label_idx = 1  # Example: Class index of the true label
num_classes = 183  # Number of classes in your model
input_label = tf.one_hot(true_label_idx, num_classes)
input_label = tf.reshape(input_label, (1, num_classes))

# Apply FGSM attack
adv_x, perturbations = apply_attack(
    model=model,
    input_image=input_image,
    input_label=input_label,
    attack_type="fgsm",
    epsilon=0.02
)

# Display and save the results
display_images(adv_x, save_path="adv_image.jpg")
display_images(perturbations, save_path="perturbations.jpg")







import tensorflow as tf
from adversarial_attacks import preprocess, apply_attack, display_images

# Load your model
model = tf.keras.models.load_model("path/to/your/model")

# Load and preprocess the input image
image_path = "/content/drive/MyDrive/Ai_Lab/data/Yazdanian/simple/51101054.jpg"
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)
input_image = preprocess(image)

# Define the true label
true_label_idx = 1  # Example: Class index of the true label
num_classes = 183  # Number of classes in your model
input_label = tf.one_hot(true_label_idx, num_classes)
input_label = tf.reshape(input_label, (1, num_classes))

# Apply FGSM attack
adv_x, perturbations = apply_attack(
    model=model,
    input_image=input_image,
    input_label=input_label,
    attack_type="fgsm",
    epsilon=0.02
)

# Display and save the results
display_images(adv_x, save_path="/content/drive/MyDrive/Ai_Lab/data/Yazdanian/yazdanian_attack_FGSM/51101054.jpg")
display_images(perturbations, save_path="/content/drive/MyDrive/Ai_Lab/data/Yazdanian/yazdanian_attack_FGSM/per_51101054.jpg")