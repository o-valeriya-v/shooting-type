# ЗДЕСЬ БУДЕТ СОБРАН ДАТАСЕТ ДЛЯ ИЗОБРАЖЕНИЙ ИЗ РАЗЛИЧНЫХ ИСТОЧНИКОВ
import numpy as np
import os
import cv2
import image
import matplotlib
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import random
from keras.preprocessing.image import img_to_array

from tensorflow.keras.models import Sequential # Сеть прямого распространения
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adadelta # оптимизаторы

## ФОРМИРУЕМ X_TRAIN И X_VAL

digital_dataset = []
thermal_dataset = []

directory_digital = '/home/valeriya_ai/PycharmProjects/shooting-type/digital/'
directory_thermal = '/home/valeriya_ai/PycharmProjects/shooting-type/thermal/'

### перечислим все файлы в каталоге и добавим их в специальные списки для дальнейшей обработки
for root, dirs, files in os.walk(directory_digital):
	for file in files:
		digital_dataset.append(os.path.join(root,file))

for root, dirs, files in os.walk(directory_thermal):
	for file in files:
		thermal_dataset.append(os.path.join(root,file))

print('Количество изображений в цифровой съемке:', len(digital_dataset))
print('Количество изображений в тепловой съемке:', len(thermal_dataset))

### Узнаем размер изображений разных датасетов
image_1 = cv2.imread(digital_dataset[100])
image_2 = cv2.imread(thermal_dataset[100])
print('Размер исходного изображения из 1 выборки', image_1.shape)
print('Размер исходного изображения из 2 выборки', image_2.shape)

### Теперь разобъем загруженные изображения на тренировочные и проверочные выборки, заранее задав единые параметры изображений
train_images = []    # Создаем пустой список для хранений оригинальных изображений обучающей выборки
y_train = []
val_images = []      # Создаем пустой список для хранений оригинальных изображений проверочной выборки
y_val = []
img_width = 256
img_height = 176


for file in digital_dataset[:1500]:
    image = cv2.imread(file)       # считываем изображение. по умолчанию в цветном формате cv2.IMREAD_COLOR
    res_img = cv2.resize(image, (img_width, img_height), cv2.INTER_AREA)
    train_images.append(res_img)


for file in thermal_dataset[:1500]:
    image = cv2.imread(file)
    res_img = cv2.resize(image, (img_width, img_height), cv2.INTER_AREA)
    train_images.append(res_img)

print('Объем обучающей выборки', len(train_images), 'изображений')

for file in digital_dataset[1500:1628]:
    image = cv2.imread(file)  # считываем изображение. по умолчанию в цветном формате cv2.IMREAD_COLOR
    res_img = cv2.resize(image, (img_width, img_height), cv2.INTER_AREA)
    val_images.append(res_img)

for file in thermal_dataset[1500:1628]:
    image = cv2.imread(file)  # считываем изображение. по умолчанию в цветном формате cv2.IMREAD_COLOR
    res_img = cv2.resize(image, (img_width, img_height), cv2.INTER_AREA)
    val_images.append(res_img)

print('Объем тестовой выборки', len(val_images), 'изображений')

x_train = np.array(train_images, dtype='int32')
x_val = np.array(val_images, dtype='int32')
print('*'*32)
print('Размер обучающей выборки', x_train.shape)
print('Размер тестовой выборки', x_val.shape)
print('*'*32)




## ФОРМИРУЕМ Y_TRAIN И Y_VAL

zero_list = [0]*1500
ones_list = [1]*1500
y_train = zero_list + ones_list
print('Размер y_train', len(y_train))

y_val = zero_list[:128] + ones_list[:128]
print('Размер y_val', len(y_val))

y_train = np.array(y_train)
y_val = np.array(y_val)
print(y_train.shape)
print(y_val.shape)


# СОЗДАЕМ НЕЙРОННУЮ СЕТЬ ДЛЯ ОБУЧЕНИЯ
# задаем batch_size
batch_size = 32

# Создаем последовательную модель
model = Sequential()
model.add(BatchNormalization(input_shape=(img_width, img_height,3))) # слой пакетной нормализации
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
# первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))  # pool_size = (2,2) означает, что картинка (нейрон) уменьшается в 2 раза
model.add(Dropout(0.25)) # отключаем 25% нейронов для следующего слоя

model.add(Flatten()) # вытягиваем данные в 1 вектор
model.add(Dense(256, activation='relu')) # сокращаем данные до 256 нейронов
model.add(Dropout(0.25)) # слой регуляризации Dropout
model.add(Dense(2, activation='softmax')) # выходной полносвязный слой

# Компилируем сеть
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = 15,
                    validation_data = (x_val, y_val),
                    verbose=1)

# Отображаем график точности обучения
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
lt.show()