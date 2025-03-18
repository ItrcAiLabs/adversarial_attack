pip install .

from .utils import preprocess, predict
from adversarial_attacks import universal_perturbation, create_poisoned_data, boundary_attack,hopskipjump_attack ,deepfool_attack
import tensorflow as tf

model = tf.keras.applications.MobileNetV2(weights='imagenet')

image = tf.keras.preprocessing.image.load_img("path/to/image.jpg")
image = tf.keras.preprocessing.image.img_to_array(image)
image = preprocess(image)

adv_image = zoo_attack(model, image, target_label=0, epsilon=0.01, max_iter=100)
print("Done")