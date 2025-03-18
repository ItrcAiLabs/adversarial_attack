# Adversarial Attacks Library

A Python library for implementing various adversarial attacks on deep learning models, built with TensorFlow. This library provides a collection of attacks such as ZOO, Boundary Attack, DeepFool, Universal Perturbation, and more, designed to test the robustness of neural networks.

## Features
- **Modular Design**: Easily extensible with new attack methods.
- **Supported Attacks**:
  - ZOO (Zeroth Order Optimization) Attack
  - Boundary Attack
  - HopSkipJump Attack
  - DeepFool Attack
  - Universal Perturbation
  - Gradient Matching / Witches' Brew (Data Poisoning)
- **Preprocessing Utilities**: Built-in functions for preparing images and predictions.
- **Customizable**: Adjustable parameters like `epsilon`, `max_iter`, and more.

## Requirements
- Python 3.6 or higher
- TensorFlow 2.0 or higher
- NumPy

## Installation
You can install the adversarial_attacks library via pip:

   ```bash
   pip install .
   ```
## Usage

Below is an example of how to use the library to perform a ZOO attack on an image using a pre-trained MobileNetV2 model.

```python
import tensorflow as tf
from adversarial_attacks import preprocess, zoo_attack

# Load a pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load and preprocess an image
image_path = "path/to/your/image.jpg"
image = tf.keras.preprocessing.image.load_img(image_path)
image = tf.keras.preprocessing.image.img_to_array(image)
image = preprocess(image)

# Perform ZOO attack
target_label = 0  # Example target label
adversarial_image = zoo_attack(
    model=model,
    input_image=image,
    target_label=target_label,
    epsilon=0.01,
    h=0.001,
    max_iter=100
)

# Display results (optional)
import matplotlib.pyplot as plt
plt.imshow(adversarial_image[0] * 0.5 + 0.5)  # Denormalize for visualization
plt.axis('off')
plt.show()
```

### Available Attacks
| Attack                | Function                  | Description                                  |
|-----------------------|---------------------------|----------------------------------------------|
| ZOO Attack            | `zoo_attack`             | Zeroth-order optimization-based attack       |
| Boundary Attack       | `boundary_attack`        | Decision boundary-based attack               |
| HopSkipJump Attack    | `hopskipjump_attack`     | Black-box attack with minimal perturbation   |
| DeepFool Attack       | `deepfool_attack`        | Efficient adversarial perturbation           |
| Universal Perturbation| `universal_perturbation` | Generates a single perturbation for all data |
| Gradient Matching     | `create_poisoned_data`   | Data poisoning attack                        |

For detailed parameters, refer to the function docstrings in the code.

## Project Structure
```
adversarial_attacks/
├── adversarial_attacks/
│   ├── __init__.py   # Package initialization
│   ├── utils.py      # Helper functions (preprocess, predict)
│   ├── attacks.py    # Attack implementations
├── setup.py          # Setup script for installation
├── README.md         # This file
```

## Contact
For questions or suggestions, feel free to open an issue or contact me at [erfanshakouri.aielec@gmail.com.](mailto:erfanshakouri.aielec@gmail.com.).
