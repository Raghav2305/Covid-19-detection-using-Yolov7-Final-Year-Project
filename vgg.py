# import necessary libraries
import tensorflow as tf
import numpy as np
from PIL import Image

# load the saved model
model = tf.keras.models.load_model('./VGG16/Best.h5')


# define the image size that the model expects
img_size = (224, 224)

# load and preprocess the input image(s)
img_path = './test_on_these_images/Normal_1.jpeg'
img = Image.open(img_path).convert('RGBA')
img = img.convert('RGB')
img = img.resize(img_size)
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# make a prediction on the input image(s)
prediction = model.predict(img_array)

# print the predicted class (assuming binary classification)
class_idx = np.argmax(prediction)
print(prediction)
print(class_idx)
if class_idx == 0:
    print('Covid-19')
else:
    print('Normal')
