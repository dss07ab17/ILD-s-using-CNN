# =========================
# FLEXIBLE SAFE CNN TRAINING
# Compatible with Windows + TensorFlow 2.x
# =========================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
import argparse
import datetime
import warnings
import tensorflow as tf
import sklearn.metrics as metrics
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, LeakyReLU
from tensorflow.keras.initializers import Orthogonal, HeUniform
from tensorflow.keras.utils import to_categorical, plot_model

warnings.filterwarnings('ignore')

target_names = ['consolidation', 'fibrosis', 'ground_glass', 'healthy', 'micronodules', 'reticulation']
today = datetime.datetime.now().strftime("_%m-%d-%Y_%H-%M")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-do', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('-a', default=0.3, type=float, help='LeakyReLU alpha')
    parser.add_argument('-k', default=4, type=int, help='Filter multiplier')
    parser.add_argument('-cl', default=7, type=int, help='Number of conv layers')
    parser.add_argument('-pt', default='Avg', choices=['Avg', 'Max', 'None'], help='Pooling type')
    parser.add_argument('-obj', default='ce', choices=['ce', 'mse'], help='Loss function')
    parser.add_argument('-epochs', default=200, type=int, help='Number of epochs')
    return parser.parse_args()

def load_data():
    X_train = pickle.load(open("./pickle/X_train.pkl", "rb"), encoding="latin1")
    y_train = pickle.load(open("./pickle/y_train.pkl", "rb"), encoding="latin1")
    X_val = pickle.load(open("./pickle/X_val.pkl", "rb"), encoding="latin1")
    y_val = pickle.load(open("./pickle/y_val.pkl", "rb"), encoding="latin1")
    
    X_train = np.expand_dims(X_train, 1) / 255.0
    X_val = np.expand_dims(X_val, 1) / 255.0

    if X_train.shape[1] == 1:
        X_train = np.transpose(X_train, (0, 2, 3, 1))
        X_val = np.transpose(X_val, (0, 2, 3, 1))

    nb_classes = np.unique(y_train).shape[0]
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)

    return (X_train, y_train), (X_val, y_val)

def load_testdata():
    X_test = pickle.load(open("./pickle/X_test.pkl", "rb"), encoding="latin1")
    y_test = pickle.load(open("./pickle/y_test.pkl", "rb"), encoding="latin1")

    X_test = np.expand_dims(X_test, 1) / 255.0

    if X_test.shape[1] == 1:
        X_test = np.transpose(X_test, (0, 2, 3, 1))

    return X_test, y_test

def get_loss(name):
    return {'ce': 'categorical_crossentropy', 'mse': 'mse'}[name]

def build_model(input_shape, output_shape, params):
    model = Sequential()
    filters = params.k * 8

    model.add(Conv2D(filters, (3, 3), padding="same", kernel_initializer=Orthogonal(), input_shape=input_shape[1:]))
    model.add(LeakyReLU(alpha=params.a))

    for i in range(2, params.cl + 1):
        if i % 4 == 0:
            filters *= 2
        model.add(Conv2D(filters, (3, 3), padding="same", kernel_initializer=Orthogonal()))
        model.add(LeakyReLU(alpha=params.a))

        if i % 4 == 0 and params.pt != 'None':
            shape = model.output_shape
            if shape[1] >= 2 and shape[2] >= 2:
                if params.pt == 'Avg':
                    model.add(AveragePooling2D(pool_size=(2, 2)))
                else:
                    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(params.do))
    model.add(Dense(512, kernel_initializer=HeUniform()))
    model.add(LeakyReLU(alpha=params.a))
    model.add(Dropout(params.do))
    model.add(Dense(256, kernel_initializer=HeUniform()))
    model.add(LeakyReLU(alpha=params.a))
    model.add(Dense(output_shape[1], activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=0.001)
    model.compile(optimizer=optimizer, loss=get_loss(params.obj), metrics=['accuracy'])
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = np.argmax(model.predict(X_val, batch_size=64), axis=1)
    y_true = np.argmax(y_val, axis=1)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    acc = metrics.accuracy_score(y_true, y_pred)
    print(f"Validation F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
    return f1, acc

def train():
    args = parse_args()

    (X_train, y_train), (X_val, y_val) = load_data()
    input_shape = X_train.shape
    output_shape = y_train.shape

    model = build_model(input_shape, output_shape, args)

    os.makedirs('./output/', exist_ok=True)
    plot_model(model, to_file='./output/model_plot.png', show_shapes=True, show_layer_names=True)
    print("Model architecture saved at ./output/model_plot.png")

    print(f"Starting training for {args.epochs} epochs...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=64, verbose=2)

    evaluate_model(model, X_val, y_val)

    os.makedirs('./pickle/', exist_ok=True)
    model.save('./pickle/flexible_cnn_model.keras')
    print("Model saved at ./pickle/flexible_cnn_model.keras")

    X_test, y_test = load_testdata()
    y_pred = np.argmax(model.predict(X_test, batch_size=64), axis=1)

    if len(y_test.shape) > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    f1_test = metrics.f1_score(y_test_labels, y_pred, average='macro')
    acc_test = metrics.accuracy_score(y_test_labels, y_pred)
    print(f"Test F1 Score: {f1_test:.4f}, Test Accuracy: {acc_test:.4f}")

    # New: Classification report and confusion matrix
    print("\nConfusion Matrix:")
    cm = metrics.confusion_matrix(y_test_labels, y_pred)
    print(cm)

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print("\nConfusion Matrix (with labels):")
    print(cm_df)

    print("\nClassification Report:")
    print(metrics.classification_report(y_test_labels, y_pred, target_names=target_names, digits=4))

if __name__ == "__main__":
    train()
