import os
import numpy as np
import cv2

#Read DNIM data set
def collect_data():
  XDataset = []
  YDataset = []

  TimePath = '/content/DNIM/time_stamp/'
  file_path = [TimePath + f for f in os.listdir(TimePath)]
  file_path.sort()
  for file in file_path:
    f = open(file, 'r')
    for line in f:
      hour = int(line.split()[-2])
      img_path = '/content/DNIM/Image/' + file.split('/')[-1][0:-4] + '/' + line.split()[0]
      XDataset.append(img_path)
      if hour > 6 and hour < 17:
        YDataset.append(0)
      else:
        YDataset.append(1)
      return (XDataset,YDataset)

#read rgb img
# resize
#normalization
def data_preprocessing(X,size):
  imgs = []
  img_size = (size, size)
  for i in X:
    img = cv2.imread(i, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype('float16')
    #normalization
    img /= 255
    imgs.append(img)
  return imgs

#split data set to train an validation
#shuffel
# return numpy arrays xt, yt, xv,yv
def split_train_valid(X,Y,ratio):
    #shuffel
    imgs = []
    labels = []
    indices_arr = np.random.permutation(len(Y))
    for i in indices_arr:
      imgs.append(X[i])
      labels.append(Y[i])

    #split
    x_len = int(ratio * len(Y))
    XTrain = []
    YTrain = []
    XValid = []
    YValid = []
    for i in range(x_len):
      XTrain.append(imgs[i])
      YTrain.append(labels[i])

    for i in range(x_len, len(Y)):
      XValid.append(imgs[i])
      YValid.append(labels[i])

    X = np.array(XTrain)
    XV = np.array(XValid)
    Y = np.array(YTrain)
    YV = np.array(YValid)

    return (X,Y,XV,YV)