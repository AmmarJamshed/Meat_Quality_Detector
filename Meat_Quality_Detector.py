#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# # Model Building

# ## Dataset splitting

# In[3]:


import os
import shutil
import random

# Paths to your original files
good_meat_dir = r'C:\Users\User\Downloads\Fresh'
spoiled_meat_dir = r'C:\Users\User\Downloads\Spoiled'


# In[5]:


# Directories for split datasets
base_dir = r'C:\Users\User\Downloads\split_data'
os.makedirs(base_dir, exist_ok=True)

for category in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_dir, category, 'good'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, category, 'spoiled'), exist_ok=True)

def split_data(source, train_size, val_size, dest_base):
    all_files = os.listdir(source)
    random.shuffle(all_files)
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size+val_size]
    test_files = all_files[train_size+val_size:]

    for f in train_files:
        shutil.copy(os.path.join(source, f), os.path.join(dest_base, 'train', 'good' if 'good' in source else 'spoiled'))
    for f in val_files:
        shutil.copy(os.path.join(source, f), os.path.join(dest_base, 'val', 'good' if 'good' in source else 'spoiled'))
    for f in test_files:
        shutil.copy(os.path.join(source, f), os.path.join(dest_base, 'test', 'good' if 'good' in source else 'spoiled'))

# Adjust numbers based on your dataset size
split_data(good_meat_dir, 70, 15, base_dir)
split_data(spoiled_meat_dir, 70, 15, base_dir)


# ## Image data generator

# In[6]:


# ImageDataGenerator for augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


# ## Load Images

# In[8]:


train_dir = r'C:\Users\User\Downloads\split_data\train'
test_dir = r'C:\Users\User\Downloads\split_data\test'
val_dir = r'C:\Users\User\Downloads\split_data\val'


# In[10]:


from tensorflow.keras.callbacks import EarlyStopping

# Adjusting the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Validation generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)


# In[16]:


print(f"Found {val_generator.samples} validation images.")


# In[17]:


validation_steps = max(1, val_generator.samples // val_generator.batch_size)
print(f"Validation steps: {validation_steps}")


# In[18]:


for x_batch, y_batch in val_generator:
    print(x_batch.shape, y_batch.shape)
    break  # Print shapes of the first batch and exit loop


# In[25]:


x_batch, y_batch = next(val_generator)
print(x_batch.shape, y_batch.shape)
print(y_batch)  # Inspect labels


# In[28]:


# Building the CNN Model
from tensorflow.keras.layers import Dropout

# Example of reducing model complexity and adding dropout for regularization
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilation step
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[29]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    verbose=1
)

# Manually evaluate on validation data
val_loss, val_accuracy = model.evaluate(val_generator, steps=max(1, val_generator.samples // val_generator.batch_size))
print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')


# In[30]:


model.save('meat_quality_model.h5')


# In[31]:


from tensorflow.keras.models import load_model

model = load_model('meat_quality_model.h5')


# # Streamlit deployment

# In[32]:


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model('meat_quality_model.h5')

# Recompile the model (if necessary)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the class names
class_names = ['Good', 'Bad']

# Title and description
st.title('Meat Quality Classifier')
st.write('Upload an image of meat, and the model will classify it as good or bad.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the image file
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image to the required input format
    img = img.resize((224, 224))  # Resize to match the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    
    # Predict
    predictions = model.predict(img_array)
    score = predictions[0]
    
    # Display the result
    st.write(f"This meat is: {class_names[int(np.round(score))]}")
    st.write(f"Confidence: {score[0]:.2f}")


# In[ ]:




