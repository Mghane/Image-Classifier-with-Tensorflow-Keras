# Create the process_image function
def process_image (image):
    import tensorflow as tf
    print('original image shape is:', image.shape)
    image = tf.cast(image, dtype=tf.float64)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    print('Processed image shape is:', image.shape)
    return image