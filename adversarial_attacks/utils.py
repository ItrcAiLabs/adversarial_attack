import tensorflow as tf
import matplotlib.pyplot as plt
from .attacks import fgsm_attack, pgd_attack, bim_attack, mi_fgsm_attack

def preprocess(image):
    """
    Preprocess the image for the model.
    
    Args:
        image: Raw image (numpy array or TensorFlow tensor)
    
    Returns:
        Preprocessed image as a TensorFlow tensor
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

def display_images(image, save_path=None):
    """
    Display and optionally save an image.
    
    Args:
        image: Image to display (TensorFlow tensor)
        save_path: Path to save the image (optional)
    """
    image = image[0] * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    if save_path:
        plt.imsave(save_path, image.numpy())
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def apply_attack(model, input_image, input_label, attack_type="fgsm", **kwargs):
    """
    Apply the specified adversarial attack to the input image.
    
    Args:
        model: Deep learning model
        input_image: Input image (TensorFlow tensor)
        input_label: Correct label (TensorFlow tensor)
        attack_type: Type of attack ("fgsm", "pgd", "bim", "mi_fgsm")
        kwargs: Parameters for the attack (e.g., epsilon, alpha, num_iterations, momentum)
    
    Returns:
        adv_x: Adversarial image
        perturbations: Perturbations applied to the input image
    """
    if attack_type == "fgsm":
        return fgsm_attack(model, input_image, input_label, kwargs.get("epsilon", 0.02))
    elif attack_type == "pgd":
        return pgd_attack(model, input_image, input_label,
                          epsilon=kwargs.get("epsilon", 0.02),
                          alpha=kwargs.get("alpha", 0.005),
                          num_iterations=kwargs.get("num_iterations", 5))
    elif attack_type == "bim":
        return bim_attack(model, input_image, input_label,
                          epsilon=kwargs.get("epsilon", 0.02),
                          alpha=kwargs.get("alpha", 0.005),
                          num_iterations=kwargs.get("num_iterations", 5))
    elif attack_type == "mi_fgsm":
        return mi_fgsm_attack(model, input_image, input_label,
                              epsilon=kwargs.get("epsilon", 0.02),
                              alpha=kwargs.get("alpha", 0.005),
                              num_iterations=kwargs.get("num_iterations", 5),
                              momentum=kwargs.get("momentum", 0.9))
    else:
        raise ValueError("attack_type must be one of: 'fgsm', 'pgd', 'bim', 'mi_fgsm'")