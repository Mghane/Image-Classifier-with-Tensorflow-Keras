import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
from processing import process_image

import argparse
import numpy as np
import json


def predict(image_path, model):
    '''
    Image object ==> top k classes predicted by model along with their probabilities 
    
    This function has three inputs: picture path, mode, and number of top k classes predicted by the model.
    It takes a picture from given path, applys necessary processing, uses the picture for prediction and returns 
    top k classes predicted by the model.
    '''
    my_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    top_k=5
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    
    ## Model accepts 4-dimension array. To satisfy this requirement we need to expand the processed image of 3 dims to 4 dims
    processed_test_image = np.expand_dims(processed_test_image,axis=0)
    print('Expanded processed image shape is:', processed_test_image.shape)
    
    ## Read Json file
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    
    ## predict
    ps = my_model.predict(processed_test_image)
    
    ## Extract probaility and class label for top k classes with highest probability
    indeces = ps.argsort()[0][::-1][:top_k].tolist() 
    probs = ps[0][indeces].tolist()
    classes = [str(i) for i in indeces]
    flowers_name = [class_names[str(x+1)] for x in indeces]  # add 1 to correspond the indexes from the prediction to the indexes in the class_names
    return probs, classes, flowers_name

parser = argparse.ArgumentParser(description = "Description for my parser")
parser.add_argument("image_path",help="Image Path", default="")
parser.add_argument("saved_model",help="Model Path", default="")   
args = parser.parse_args()

probs, classes, flowers_name = predict(args.image_path, args.saved_model)
print('probability distribution is:\n', probs)
print('for label numbers of:\n', classes)
print('and flower names of:\n',flowers_name)
