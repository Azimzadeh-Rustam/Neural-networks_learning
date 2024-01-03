# Распознавание рукописных цифр

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(images_train, numbers_train), (images_test, numbers_test) = mnist.load_data()

# Стандартизация входных данных (для того чтобы входные данные принимали значения от 0 до 1)
images_train = images_train / 255
images_test = images_test / 255

numbers_train_cat = keras.utils.to_categorical(numbers_train, 10)
numbers_test_cat = keras.utils.to_categorical(numbers_test, 10)

# Отображаем первые 25 изображений из обучающей выборки
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images_train[i], cmap = plt.cm.binary)

plt.show()

# Формирование модели нейронной сети и вывод ее структуры в консоль
model = keras.Sequential([
    Flatten(input_shape = (28, 28, 1)),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

print(model.summary())

# Компиляция НС с оптимизацией по Adam и критерием - категориальная кросс-энтропия
model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

# Другие оптимизаторы или свой оптимизатор
#myAdam = keras.opimizers.Adam(learning_rate = 0.1)
#myOpt = keras.opimizers.SGD(learning_rate = 0.1, momentum = 0.0, nesterov = True)
#model.compile(optimizer = myOpt,
#             loss = 'categorical_crossentropy',
#             metrics = ['accuracy'])

# Запуск процесса обучения: 80% - обучающая выборка, 20% - выборка валидации
model.fit(images_train, numbers_train_cat, batch_size = 32, epochs = 5, validation_split = 0.2)

# Тестируем НС
model.evaluate(images_test, numbers_test_cat)

# Для теста подадим какое-то конкретное изображение и проверим как НС с ней справится
n = 0
image = np.expand_dims(images_test[n], axis = 0)
number = model.predict(image)
print(number)
print(f"Распознанная цифра: {np.argmax(number)}")

plt.imshow(images_test[n], cmap = plt.cm.binary)
plt.show()