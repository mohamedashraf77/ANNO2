from keras import models
import cv2
import numpy as np

def predict(img):
  model = models.load_model("/content/model.h5")
  model.summary()
  img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
  img = img.astype('float16')
  img /= 255
  x_input = []
  x_input.append(img)
  sample = np.array(x_input)
  prediction = model.predict(sample)
  print(prediction)
  return prediction