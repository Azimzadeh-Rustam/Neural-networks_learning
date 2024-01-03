# F = C * 1.8 + 32

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

input_layer = np.array([-40, -10, 0, 8, 15, 22, 38]) # C
output_layer = np.array([-40, 14, 32, 46, 59, 72, 100]) # F

# Создаем экземпляр класса Sequential, который создает модель многослойной нейронной сети
model = keras.Sequential()
# С помощью класса Dense добавляем в модель первый слой нейронов
model.add(Dense(units = 1, input_shape = (1,), activation = 'linear'))
model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.1))

# Обучаем нейронную сеть
history = model.fit(input_layer, output_layer, epochs = 500, verbose = False)

# Рисуем график критерия качества для каждой эпохи обучения
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

# Выводим предсказание нейросети после обучения
print(model.predict([100]))

# Выводим полученные весовые коэффициенты у нейронной сети
print(model.get_weights())
