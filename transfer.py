# -*- coding: utf-8 -*-
"""
Created on Wed May 14 01:49:58 2025

@author: Mohannad
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing import image

# Paths to dataset folders (updated to your user directory)
train_dir = 'C:/Users/Mohannad/Desktop/kodlar/cats_and_dogs_dataset/train'
validation_dir = 'C:/Users/Mohannad/Desktop/kodlar/cats_and_dogs_dataset/validation'

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20
LEARNING_RATE = 1e-5

# Data generators with normalization (rescale pixel values to [0,1])
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load validation dataset
val_ds = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load pretrained MobileNetV2 model without top layer, freeze weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build the classification head on top of base model
x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Plot training & validation accuracy values
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Predict on a single image example
img_path = r'C:\Users\Mohannad\Desktop\kodlar\cats_and_dogs_dataset\kedi.jpeg'  # Image file path
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

predictions = model.predict(img_array)
class_names = ['Cat', 'Dog']

predicted_class = class_names[np.argmax(predictions[0])]
print("Prediction result:", predicted_class)

for i, prob in enumerate(predictions[0]):
    print(f"{class_names[i]}: {prob * 100:.2f}%")

# Display the image with prediction
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.title(f'{predicted_class}, Probability={predictions[0][np.argmax(predictions[0])]:.3f}', fontsize=16)
plt.tight_layout()
plt.show()
