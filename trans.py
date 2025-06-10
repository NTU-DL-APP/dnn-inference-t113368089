import tensorflow as tf
import numpy as np
import json

# 1. 載入模型
model = tf.keras.models.load_model('model/fashion_mnist.h5')

# 2. 轉換成簡化格式的 architecture
architecture = []
for layer in model.layers:
    ltype = layer.__class__.__name__

    if ltype == "Dropout":
        continue  # 推論不需要 Dropout

    entry = {
        "name": layer.name,
        "type": ltype,
        "weights": [],
        "config": {}
    }
    if ltype == "Dense":
        entry["weights"] = [f"{layer.name}_W", f"{layer.name}_b"]
        entry["config"]["activation"] = layer.activation.__name__
    elif ltype == "Flatten":
        entry["weights"] = []
        entry["config"] = {}

    architecture.append(entry)

# 3. 輸出 JSON
with open('model/fashion_mnist.json', 'w') as f:
    json.dump(architecture, f, indent=2)

# 4. 儲存權重
weights = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        W, b = layer.get_weights()
        weights[f"{layer.name}_W"] = W
        weights[f"{layer.name}_b"] = b

np.savez('model/fashion_mnist.npz', **weights)