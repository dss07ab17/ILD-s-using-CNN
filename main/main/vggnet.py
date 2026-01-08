import os
import cv2
import pickle
import datetime
import numpy as np
import argparse
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# --- Timestamp ---
t = datetime.datetime.now()
today = f'_{t.month}-{t.day}-{t.year}_{t.hour}:{t.minute}'

# --- CLI Args ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-do', default=0.5, type=float)
    parser.add_argument('-pat', default=30, type=int)
    parser.add_argument('-obj', default='ce')
    parser.add_argument('-csv', default='res')
    return parser.parse_args()

# --- Load Pickle Data ---
def load_pickle_data(path):
    return pickle.load(open(path, "rb"), encoding="latin1")

def load_data():
    X_train = load_pickle_data("/content/drive/MyDrive/main/pickle/X_train.pkl")
    y_train = load_pickle_data("/content/drive/MyDrive/main/pickle/y_train.pkl")
    X_val = load_pickle_data("/content/drive/MyDrive/main/pickle/X_val.pkl")
    y_val = load_pickle_data("/content/drive/MyDrive/main/pickle/y_val.pkl")
    nb_classes = len(np.unique(y_train))
    zbn = np.min(y_train)
    y_train = to_categorical(y_train - zbn, nb_classes)
    y_val = to_categorical(y_val - zbn, nb_classes)
    return (X_train, y_train), (X_val, y_val)

def load_testdata():
    X_test = load_pickle_data("/content/drive/MyDrive/main/pickle/X_test.pkl")
    y_test = load_pickle_data("/content/drive/MyDrive/main/pickle/y_test.pkl")
    return np.asarray(X_test), y_test

def convert_to_rgb_and_resize(X, size=(128, 128)):
    return np.array([np.stack([cv2.resize(img.squeeze(), size)] * 3, axis=-1).astype(np.float32) for img in X])

# --- Evaluation ---
def evaluate(actual, pred):
    return (
        metrics.f1_score(actual, pred, average='macro'),
        metrics.accuracy_score(actual, pred),
        metrics.confusion_matrix(actual, pred)
    )

# --- Save / Load Model ---
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

# --- VGG19 Fine-tuned Model ---
def get_vgg_model(input_shape, output_shape, dropout):
    base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers except block5
    for layer in base.layers:
        layer.trainable = False
        if 'block5' in layer.name:
            layer.trainable = True

    x = base.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(output_shape[1], activation='softmax')(x)
    return Model(inputs=base.input, outputs=predictions)

# --- Loss ---
def get_loss(obj):
    return {'mse': 'mse', 'ce': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)}.get(obj, 'categorical_crossentropy')

# --- Training ---
def train(X_train, y_train, X_val, y_val, dropout, epochs, loss_func, class_weights):
    model = get_vgg_model(X_train.shape[1:], y_train.shape, dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=loss_func, metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_val, y_val),
              epochs=epochs,
              class_weight=class_weights,
              callbacks=callbacks,
              verbose=1)

    y_pred = model.predict(X_val, batch_size=64)
    f1, acc, _ = evaluate(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
    print(f'Validation F1: {f1:.4f}, Accuracy: {acc:.4f}')
    store_model(model)
    return model

# --- Testing ---
def test(X_test, y_test, loss_func):
    model = load_model_file()
    model.compile(optimizer='Adam', loss=loss_func, metrics=['accuracy'])
    X_test = preprocess_input(X_test)
    y_pred = model.predict(X_test, batch_size=64)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    f1, acc, cm = evaluate(y_true, np.argmax(y_pred, axis=1))
    print(f'Test F1: {f1:.4f}, Accuracy: {acc:.4f}\nConfusion Matrix:\n{cm}')

# --- Main ---
if __name__ == "__main__":
    args = parse_args()
    dropout, epochs, loss_func = args.do, args.pat, get_loss(args.obj)
    run_id = args.csv + today

    (X_train, y_train), (X_val, y_val) = load_data()
    X_train = preprocess_input(convert_to_rgb_and_resize(X_train))
    X_val = preprocess_input(convert_to_rgb_and_resize(X_val))

    labels = np.argmax(y_train, axis=1)
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}

    model = train(X_train, y_train, X_val, y_val, dropout, epochs, loss_func, class_weight_dict)

    X_test, y_test = load_testdata()
    X_test = preprocess_input(convert_to_rgb_and_resize(X_test))
    test(X_test, y_test, loss_func)

    import gc
    tf.keras.backend.clear_session()
    del model
    gc.collect()
