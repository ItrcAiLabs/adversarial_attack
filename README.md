# Adversarial Attacks Library

![Python](https://img.shields.io/badge/python-3.6%2B-blue)

A Python library for applying adversarial attacks on deep learning models using TensorFlow. This library provides a variety of attack methods, including FGSM, PGD, BIM, and MI-FGSM, to create adversarial examples for testing the robustness of neural networks.

## Features

- **Multiple Adversarial Attacks:** Supports FGSM, PGD, BIM, and MI-FGSM attacks.
- **Flexible Interface:** Easily apply attacks on custom TensorFlow models with configurable parameters.
- **Preprocessing and Visualization Utilities:** Includes functions for preprocessing images and visualizing adversarial examples.
- **Batch Processing:** Apply attacks to single images or integrate into larger pipelines.

## Installation

You can install the `adversarial_attacks` library via pip:

```bash
pip install adversarial_attacks
```

### Prerequisites

Ensure you have the following dependencies installed:
- Python 3.6 or higher
- `tensorflow>=2.0.0`
- `matplotlib>=3.0.0`
- `numpy>=1.19.0`

These dependencies will be automatically installed when you install the package via pip.

## Usage

The library provides a main function `apply_attack` to apply adversarial attacks on images, along with helper functions for preprocessing and visualization. Below are some examples of how to use the library:

### Example 1: Apply FGSM Attack

Apply the Fast Gradient Sign Method (FGSM) attack on an image to create an adversarial example:

```python
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
```



## Available Attacks

The library supports the following adversarial attack methods through the `apply_attack` function:

| Attack Type | Description                              | Required Parameters                     | Optional Parameters                     |
|-------------|------------------------------------------|-----------------------------------------|-----------------------------------------|
| `fgsm`      | Fast Gradient Sign Method               | `model`, `input_image`, `input_label`, `epsilon` | -                                       |
| `pgd`       | Projected Gradient Descent              | `model`, `input_image`, `input_label`, `epsilon` | `alpha`, `num_iterations`              |
| `bim`       | Basic Iterative Method                  | `model`, `input_image`, `input_label`, `epsilon` | `alpha`, `num_iterations`              |
| `mi_fgsm`   | Momentum Iterative FGSM                 | `model`, `input_image`, `input_label`, `epsilon` | `alpha`, `num_iterations`, `momentum`  |


## Project Structure

```
adversarial_attacks/
│
├── adversarial_attacks/      # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── attacks.py            # Core adversarial attack functions
│   ├── utils.py              # Helper functions (preprocessing, visualization)
│   ├── data_preprocessing.py            # COCO data preprocessing
│   ├── data_preprocessing_test.py       # prepper data
│   ├── models.py            # Simple CNN2D model

│
├── tests/                    # Test directory (optional)
│   ├── test.py       # Unit tests
│
├── setup.py                  # Setup script for installation
├── README.md                 # Project documentation
```



## Acknowledgments

- This library is built on top of [TensorFlow](https://www.tensorflow.org/) for deep learning model integration.
- The attack methods are inspired by research in adversarial machine learning, including FGSM, PGD, BIM, and MI-FGSM.

## Contact

For questions or feedback, you can reach out to [erfanshakouri.aielec@gmail.com](mailto:erfanshakouri.aielec@gmail.com).
