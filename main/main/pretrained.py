# --- Final Optimized AlexNet-Only Script for Google Colab ---

import os
import cv2
import pickle
import datetime
import numpy as np
import argparse
import sklearn.metrics as metrics
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential, model_from_json
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                          BatchNormalization, ReLU)
from keras.utils import to_categorical
from keras.initializers import HeNormal

# --- Timestamp for file versioning ---
t = datetime.datetime.now()
today = f'_{t.month}-{t.day}-{t.year}_{t.hour}:{t.minute}'

# --- CLI Arguments ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-do', default=0.5, type=float)
    parser.add_argument('-pat', default=20, type=int)
    parser.add_argument('-obj', default='ce')
    parser.add_argument('-csv', default='res')
    return parser.parse_args()

# --- Data Loaders ---
def load_pickle_data(path):
    return pickle.load(open(path, "rb"), encoding="latin1")

def load_data():
    X_train = load_pickle_data("/content/drive/MyDrive/main/pickle/X_train.pkl")
    y_train = load_pickle_data("/content/drive/MyDrive/main/pickle/y_train.pkl")
    X_val = load_pickle_data("/content/drive/MyDrive/main/pickle/X_val.pkl")
    y_val = load_pickle_data("/content/drive/MyDrive/main/pickle/y_val.pkl")

    X_train = np.asarray(X_train) / 255.
    X_val = np.asarray(X_val) / 255.

    nb_classes = len(np.unique(y_train))
    zbn = np.min(y_train)
    y_train = to_categorical(y_train - zbn, nb_classes)
    y_val = to_categorical(y_val - zbn, nb_classes)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    return (X_train, y_train), (X_val, y_val)

def load_testdata():
    X_test = load_pickle_data("/content/drive/MyDrive/main/pickle/X_test.pkl")
    y_test = load_pickle_data("/content/drive/MyDrive/main/pickle/y_test.pkl")
    return np.asarray(X_test) / 255., y_test

def convert_to_rgb_and_resize(X, target_size=(128, 128)):
    return np.array([np.stack([cv2.resize(img.squeeze(), target_size)] * 3, axis=-1).astype(np.float32) for img in X])

# --- Metrics ---
def evaluate(actual, pred):
    return (
        metrics.f1_score(actual, pred, average='macro'),
        metrics.accuracy_score(actual, pred),
        metrics.confusion_matrix(actual, pred)
    )

# --- Model Saving/Loading ---
def store_model(model):
    os.makedirs('./pickle', exist_ok=True)
    with open('/content/drive/MyDrive/main/pickle/ILD_CNN_model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('/content/drive/MyDrive/main/pickle/ILD_CNN_model_weights.weights.h5')

def load_model_file():
    with open('/content/drive/MyDrive/main/pickle/ILD_CNN_model.json') as f:
        model = model_from_json(f.read())
    model.load_weights('/content/drive/MyDrive/main/pickle/ILD_CNN_model_weights.weights.h5')
    return model

def get_Obj(obj):
    return {'mse': 'MSE', 'ce': 'categorical_crossentropy'}[obj]

# --- AlexNet Model ---
def get_alexnet_model(input_shape, output_shape, dropout):
    model = Sequential()
    model.add(Conv2D(64, (11, 11), strides=(4, 4), padding='valid', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(BatchNormalization()); model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (5, 5), padding="same", kernel_initializer='he_normal'))
    model.add(BatchNormalization()); model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
    model.add(BatchNormalization()); model.add(ReLU())

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
    model.add(BatchNormalization()); model.add(ReLU())

    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal'))
    model.add(BatchNormalization()); model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal')); model.add(ReLU()); model.add(Dropout(dropout))
    model.add(Dense(1024, kernel_initializer='he_normal')); model.add(ReLU()); model.add(Dropout(dropout))
    model.add(Dense(output_shape[1], activation='softmax'))
    return model

# --- Training ---
def CNN_Train(x_train, y_train, x_val, y_val, dropout, epochs, loss_func, class_weights):
    model = get_alexnet_model(x_train.shape[1:], y_train.shape, dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.001),
                  loss=get_Obj(loss_func), metrics=['accuracy'])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda e: 1e-3 * (0.95 ** e))
    
    print('Starting training...')
    model.fit(x_train, y_train, batch_size=32, epochs=epochs,
              validation_data=(x_val, y_val), class_weight=class_weights,
              callbacks=[lr_schedule], verbose=1)

    y_score = model.predict(x_val, batch_size=64)
    fscore, acc, _ = evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))
    print(f'Training done. Final fscore: {fscore:.4f}, acc: {acc:.4f}')
    store_model(model)
    return model

# --- Testing ---
def CNN_Prediction(X_test, y_test, loss_func):
    model = load_model_file()
    model.compile(optimizer='Adam', loss=get_Obj(loss_func), metrics=['accuracy'])
    y_score = model.predict(X_test, batch_size=64)
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    fscore, acc, cm = evaluate(y_true, y_pred)
    print(f'Test fscore: {fscore:.4f}\nTest accuracy: {acc:.4f}\nConfusion matrix:\n{cm}')

# --- Main Flow ---
if __name__ == "__main__":
    args = parse_args()
    dropout, epochs, loss_func = args.do, args.pat, args.obj
    res_alias = args.csv + today

    (X_train, y_train), (X_val, y_val) = load_data()
    X_train = convert_to_rgb_and_resize(X_train)
    X_val = convert_to_rgb_and_resize(X_val)

    classes = np.unique(np.argmax(y_train, axis=1))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.argmax(y_train, axis=1))
    class_weights_dict = {cls: w for cls, w in zip(classes, weights)}

    model = CNN_Train(X_train, y_train, X_val, y_val, dropout, epochs, loss_func, class_weights_dict)

    X_test, y_test = load_testdata()
    X_test = convert_to_rgb_and_resize(X_test)
    CNN_Prediction(X_test, y_test, loss_func)

    import gc
    tf.keras.backend.clear_session()
    del model
    gc.collect()
