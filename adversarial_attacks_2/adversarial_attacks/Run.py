###RUN####
########### Run Gradient Matching / Witches' Brew Attack
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255.0
target_gradient = tf.random.normal(shape=(28, 28, 1))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
poisoned_train_dataset = create_poisoned_data(train_dataset, target_gradient)
################################
# Example of using Universal Adversarial Perturbation attack
if __name__ == "__main__":
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    image_paths = ["/content/drive/MyDrive/51101002.JPG","/content/drive/MyDrive/51101001.JPG", "/content/drive/MyDrive/51101054.jpg"]
    images = [tf.keras.preprocessing.image.load_img(path) for path in image_paths]
    images = [tf.keras.preprocessing.image.img_to_array(image) for image in images]
    images = [preprocess(image) for image in images]

    perturbation = universal_perturbation(model, images, max_iter=100, epsilon=0.01, delta=0.1)
    perturbation = perturbation[0] * 0.5 + 0.5  
    plt.imshow(perturbation)
    plt.axis('off')
    plt.show()
    sample_image = images[0]
    perturbed_image = sample_image + perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, -1, 1) 
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(sample_image[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Perturbed Image")
    plt.imshow(perturbed_image[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.show()
################################
if __name__ == "__main__":
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    image_path = "/content/drive/MyDrive/Ai_Lab/data/Yazdanian/simple/51101054.jpg"
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = preprocess(image)

    target_label = 0  
    # Example of using a Zeroth order optimization (ZOO)   
    adversarial_image = zoo_attack(model, image, target_label, epsilon=0.01, h=0.001, max_iter=100)
    
    # Example of using a Boundary Attack   
    adversarial_image = boundary_attack(model, image, target_label, max_iter=1000, epsilon=0.01)
    
    # Example of using HopSkipJumpAttack   
    adversarial_image = hopskipjump_attack(model, image, target_label, max_iter=100, epsilon=0.01, delta=0.01)
    
    # Example of using DeepFool attack
    adversarial_image = deepfool_attack(model, image, max_iter=50, overshoot=0.02)

    adversarial_image = adversarial_image[0] * 0.5 + 0.5 
    plt.imshow(adversarial_image)
    plt.axis('off')
    plt.show()