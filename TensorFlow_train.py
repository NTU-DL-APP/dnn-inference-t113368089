import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# 1. 加载并归一化数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 搭建模型
model = Sequential([
    Flatten(input_shape=(28,28), name='flatten'),
    Dense(128, activation='relu', name='dense_1'),
    Dropout(0.3, name='dropout_1'),
    Dense(10,  activation='softmax', name='dense_2')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 训练
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,
          validation_split=0.1)

# 4. 保存为 .h5
model.save('model/fashion_mnist.h5')

# 5. 导出 architecture (.json)
model_json = model.to_json()
with open('model/fashion_mnist.json', 'w') as f:
    f.write(model_json)

# 6. 导出 weights (.npz)
#    keys 要和 json 中 layer['weights'] 一一对应
layer_names = ['dense_1', 'dense_2']
weight_dict = {}
for name in layer_names:
    layer = model.get_layer(name)
    W, b = layer.get_weights()
    weight_dict[f"{name}_W"] = W
    weight_dict[f"{name}_b"] = b
np.savez('model/fashion_mnist.npz', **weight_dict)