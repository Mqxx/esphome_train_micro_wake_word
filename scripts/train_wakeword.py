import argparse
import numpy as np
import tensorflow as tf
import os

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(8, 3, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 16, 1).astype(np.float32)
        yield [data]

def estimate_tensor_arena_size(tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter._get_arena_used_bytes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake_word", required=True)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    X = np.random.rand(100, 16, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(100,))
    y = tf.keras.utils.to_categorical(y, num_classes=2)

    model = build_model((16, 1), 2)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=2, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open("output/model.tflite", "wb") as f:
        f.write(tflite_model)

    arena_size = estimate_tensor_arena_size(tflite_model)
    with open("output/tensor_arena_size.txt", "w") as f:
        f.write(str(arena_size))
