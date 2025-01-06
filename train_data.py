import os
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'C:/Users/cnkum/OneDrive/Documents/Medical_Diagnosis-main/Dataset/dataset/train'
valid_path = 'C:/Users/cnkum/OneDrive/Documents/Medical_Diagnosis-main/Dataset/dataset/test'

# don't train existing weights
vgg = tf.keras.applications.vgg19.VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob(os.path.join(train_path, '*'))

# our layers - you can add more if you want
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

try:
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory(valid_path,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')
except Exception as e:
    print(f"Error loading data: {e}")
    training_set = None
    test_set = None

if training_set and test_set:
    # fit the model
    r = model.fit(
      training_set,
      validation_data=test_set,
      epochs=10,
      steps_per_epoch=len(training_set),
      validation_steps=len(test_set)
    )

    # loss
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    # plot the accuracy
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')

    # save the model
    model.save('CardioGPT.h5')
else:
    print("Failed to load training or validation data.")
