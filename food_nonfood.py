import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np

# %%
x_train = []
y_train = []

i = 0

for file in glob.glob('TRAIN/*/*.jpg'):
    if i % 1000 == 0:
        print(i)
    class_image = int(file.split('\\')[1])
    img = cv2.imread(file)
    x_train.append(img)
    y_train.append(class_image)
    i += 1

# %%
i = 0
x_test = []
y_test = []

for file in glob.glob('TEST/*/*.jpg'):
    if i % 1000 == 0:
        print(i)
    class_image = int(file.split('\\')[1])
    img = cv2.imread(file)
    x_test.append(img)
    y_test.append(class_image)
    i += 1

# %%
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

plt.imshow(cv2.cvtColor(x_train[2022], cv2.COLOR_BGR2RGB))
print(y_train[2022])

# %%
plt.imshow(cv2.cvtColor(x_train[10000], cv2.COLOR_BGR2RGB))
print(y_train[10000])

joblib.dump(x_train, 'x_train')
joblib.dump(y_train, 'y_train')
joblib.dump(x_test, 'x_test')
joblib.dump(x_test, 'y_test')

# %%
img_height = 240
img_width = 320
bs = 8
means = np.array([98, 112, 128])

train_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True,
                                   rotation_range=30, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)

train_datagen.mean = np.array(means/255, dtype=np.float32).reshape(1, 1, 3)

train_generator = train_datagen.flow_from_directory('TRAIN', target_size=(
    img_height, img_width), batch_size=bs, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True,
                                  rotation_range=30, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)

test_datagen.mean = np.array(means/255, dtype=np.float32).reshape(1, 1, 3)

test_generator = test_datagen.flow_from_directory('TEST', target_size=(
    img_height, img_width), batch_size=bs, class_mode='binary')


# %%
classifier = Sequential()
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),
               input_shape=(img_height, img_width, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
print(classifier.summary())

# %%
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

classifier.compile(optimizer=optimizer,
                   loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss',
                             save_best_only=True, save_weights_only=False, verbose=1, mode='auto')

steps = len(train_generator)
h = classifier.fit(train_generator, validation_data=test_generator,
                   steps_per_epoch=steps, epochs=10, callbacks=[checkpoint])

# %%


def scheduler(epoch, learning_rate):
    if epoch < 2:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


steps = 500


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = classifier.fit(train_generator, validation_data=test_generator,
                         steps_per_epoch=steps, epochs=5, callbacks=[checkpoint, callback])


# %%
loaded_model = load_model('model.h5')

y_pred = loaded_model.predict(test_generator)
pred_labels = np.zeros_like(y_pred)
pred_labels[y_pred < 0.5] = 0
pred_labels[y_pred >= 0.5] = 1
gt = test_generator.labels

accuracy = accuracy_score(gt, pred_labels)
precision = precision_score(gt, pred_labels)
recall = recall_score(gt, pred_labels)

print(accuracy)
print(precision)
print(recall)

# %%
resnet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(img_height, img_width, 3), pooling='max')

freezas = len(resnet50.layers) - 26

for layer in resnet50.layers[:freezas]:
    layer.trainable = False
for layer in resnet50.layers[freezas:]:
    layer.trainable = True

prediction_layer = Dense(units=1, activation='sigmoid')(resnet50.output)

model = Model(resnet50.input, prediction_layer)


# %%
y_pred = model.predict(test_generator)
pred_labels = np.zeros_like(y_pred)
pred_labels[y_pred < 0.5] = 0
pred_labels[y_pred >= 0.5] = 1
gt = test_generator.labels

accuracy = accuracy_score(gt, pred_labels)
precision = precision_score(gt, pred_labels)
recall = recall_score(gt, pred_labels)

print(accuracy)
print(precision)
print(recall)
