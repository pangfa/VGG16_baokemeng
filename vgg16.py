import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

pokemon_train = np.load(r"C:\Users\sen\Desktop\pokemon_train.npy")
pokemon_test = np.load(r"C:\Users\sen\Desktop\pokemon_test.npy")
x_train = pokemon_train[:, 1:].reshape(-1, 128, 128, 3)
y_train = pokemon_train[:, 0].reshape([-1])
x_test = pokemon_test.reshape(-1, 128, 128, 3)

# 可视化前10个训练数据
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(x_train[i])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()

print('这十张图片的标签分别是：', y_train[:10])

# 将标签对应为宝可梦种类
label_name = {0:'妙蛙种子', 1:'小火龙', 2:'超梦', 3:'皮卡丘', 4:'杰尼龟'}
name_list = []
for i in range(10):
    name_list.append(label_name[y_train[i]])
print('这十张图片标签对应的宝可梦种类分别为：', name_list)


import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.models import load_model

x_train = pokemon_train[:, 1:].reshape(-1, 128, 128, 3)
y_train = pokemon_train[:, 0].reshape([-1])
#x_test = pokemon_test.reshape(-1, 128, 128, 3)

x_train = x_train / 255
y_train = keras.utils.to_categorical(y_train)
x_test = x_train[900:]
x_train = x_train[:900]
y_test = y_train[900:]
y_train = y_train[:900]
random_seed = 2
x_train, x_val, y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,
                                                  random_state=random_seed)
from keras.applications.vgg16 import VGG16
con_base = VGG16(weights='imagenet',
                 include_top = False,
                 input_shape=(128,128,3))
print(con_base.summary())

model = Sequential()
model.add(con_base)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation="softmax"))
con_base.trainable = True
set_trainable = False
for layer in con_base.layers:
    if layer.name =='block_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

opt = RMSprop(lr=0.00001,rho=0.9,decay=0)
model.compile(loss="categorical_crossentropy",  optimizer=opt, metrics=['accuracy'])
                            
#model.fit(x_train, y_train, batch_size=32, epochs=20)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,
                                            factor=0.5,min_lr=0.00001)
epochs=50
batch_size = 16

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images)

datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                              epochs=epochs,validation_data=(x_val,y_val),
                              verbose=2,steps_per_epoch=x_train.shape[0]//batch_size,
                              callbacks =[learning_rate_reduction])


score = model.evaluate(x_test,y_test,verbose=0)
print('损失:',score[0])
print('模型预测准确率为: ',score[1])
