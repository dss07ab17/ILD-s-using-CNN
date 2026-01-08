# --- Imports ---
import re
import os
import sys
import cv2
import pickle
import datetime
import numpy as np
import argparse
import sklearn.metrics as metrics

import keras
from keras.models import Sequential
from keras.models import load_model as keras_load_model, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import Orthogonal, HeUniform

from sklearn.utils.class_weight import compute_class_weight

# --- Time for naming files ---
t = datetime.datetime.now()
today = str('_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+':'+str(t.minute))

# --- Helper functions ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-do', help='Dropout param [default: 0.5]')
    parser.add_argument('-a', help='Conv Layers LeakyReLU alpha param [default: 0.3]')
    parser.add_argument('-k', help='Feature maps k multiplier [default: 4]')
    parser.add_argument('-cl', help='Number of Convolutional Layers [default: 5]')
    parser.add_argument('-s', help='Input Image rescale factor [default: 1]')
    parser.add_argument('-pf', help='Percentage of pooling layer: [0,1] [default: 1]')
    parser.add_argument('-pt', help='Pooling type: Avg or Max [default: Avg]')
    parser.add_argument('-fp', help='Feature maps policy: proportional or static [default: proportional]')
    parser.add_argument('-opt', help='Optimizer: SGD, Adagrad, Adam [default: Adam]')
    parser.add_argument('-obj', help='Objective: mse, ce [default: ce]')
    parser.add_argument('-pat', help='Patience for early stopping [default: 200]')
    parser.add_argument('-tol', help='Tolerance for early stopping [default: 1.005]')
    parser.add_argument('-csv', help='CSV results filename alias [default: res]')
    parser.add_argument('-modeltype', help="Model type: 'custom' or 'lenet' [default: custom]", default='custom')
    args = parser.parse_args()
    return args

def load_data():
    X_train = pickle.load(open("./pickle/X_train.pkl", "rb"), encoding="latin1")
    y_train = pickle.load(open("./pickle/y_train.pkl", "rb"), encoding="latin1")
    X_val = pickle.load(open("./pickle/X_val.pkl", "rb"), encoding="latin1")
    y_val = pickle.load(open("./pickle/y_val.pkl", "rb"), encoding="latin1")

    X_train = np.asarray(np.expand_dims(X_train, 1)) / 255.
    X_val = np.asarray(np.expand_dims(X_val, 1)) / 255.

    uniquelbls = np.unique(y_train)
    nb_classes = uniquelbls.shape[0]
    zbn = np.min(uniquelbls)
    y_train = to_categorical(y_train - zbn, nb_classes)
    y_val = to_categorical(y_val - zbn, nb_classes)

    return (X_train, y_train), (X_val, y_val)

def load_testdata():
    X_test = pickle.load(open("./pickle/X_test.pkl", "rb"), encoding="latin1")
    y_test = pickle.load(open("./pickle/y_test.pkl", "rb"), encoding="latin1")

    X_test = np.asarray(np.expand_dims(X_test, 1)) / 255.
    return (X_test, y_test)

def evaluate(actual, pred):
    fscore = metrics.f1_score(actual, pred, average='macro')
    acc = metrics.accuracy_score(actual, pred)
    cm = metrics.confusion_matrix(actual, pred)
    return fscore, acc, cm

def store_model(model):
    json_string = model.to_json()
    os.makedirs('./pickle', exist_ok=True)
    open('./pickle/ILD_CNN_model.json', 'w').write(json_string)
    model.save_weights('./pickle/ILD_CNN_model_weights.weights.h5')
    return json_string

def load_model_file():
    model = model_from_json(open('./pickle/ILD_CNN_model.json').read())
    model.load_weights('./pickle/ILD_CNN_model_weights.weights.h5')
    return model

def get_FeatureMaps(L, policy, constant=17):
    return {'proportional': (L+1)**2, 'static': constant}[policy]

# 🔥 CORRECTED get_Obj
def get_Obj(obj):
    return {'mse': 'MSE', 'ce': 'categorical_crossentropy'}[obj]

# --- Model Building ---
def get_model(input_shape, output_shape, params):
    print(f"Compiling model ({params['modeltype']})...")
    model = Sequential()

    if params['modeltype'] == 'lenet':
        # --- Strict LeNet-5 Architecture ---
        model.add(Conv2D(6, (5, 5), activation='relu', input_shape=input_shape[1:], padding='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (5, 5), activation='relu', padding='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(output_shape[1], activation='softmax'))

    else:
        # --- Custom CNN ---
        fm_size = input_shape[-1] - params['cl']
        model.add(Conv2D(params['k']*get_FeatureMaps(1, params['fp']), (2,2), kernel_initializer=Orthogonal(), padding="SAME", input_shape=input_shape[1:]))
        model.add(LeakyReLU(alpha=params['a']))

        for i in range(2, params['cl']+1):
            model.add(Conv2D(params['k']*get_FeatureMaps(i, params['fp']), (2,2), kernel_initializer=Orthogonal()))
            model.add(LeakyReLU(alpha=params['a']))

        input_shape = model.output_shape[1:3]
        pool_siz = min(input_shape[0], input_shape[1])
        model.add(AveragePooling2D(pool_size=(pool_siz, pool_siz)))

        model.add(Flatten())
        model.add(Dropout(params['do']))

        model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp'])/params['pf']*6), kernel_initializer=HeUniform()))
        model.add(LeakyReLU(alpha=params['a']))
        model.add(Dropout(params['do']))

        model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp'])/params['pf']*2), kernel_initializer=HeUniform()))
        model.add(LeakyReLU(alpha=params['a']))
        model.add(Dropout(params['do']))

        model.add(Dense(output_shape[1], kernel_initializer=HeUniform(), activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=0.001)
    model.compile(optimizer=optimizer, loss=get_Obj(params['obj']), metrics=['accuracy'])
    return model

# --- Training ---
def CNN_Train(x_train, y_train, x_val, y_val, params, class_weights):
    parameters_str = '_model_' + params['modeltype']
    model = get_model(x_train.shape, y_train.shape, params)

    maxf = maxacc = maxit = 0
    best_model = model
    it = 0
    p = 0

    os.makedirs('./output/', exist_ok=True)

    print('Starting training...')
    while p < params['patience']:
        p += 1
        print('Epoch:', it)

        history = model.fit(x_train, y_train, batch_size=250, epochs=1, validation_data=(x_val, y_val), shuffle=True, class_weight=class_weights)

        y_score = model.predict(x_val, batch_size=1050)
        fscore, acc, cm = evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))

        if fscore > maxf * params['tolerance']:
            best_model = model
            maxf, maxacc, maxit = fscore, acc, it
            store_model(best_model)

        it += 1

    print('Training done. Best fscore:', maxf, 'acc:', maxacc)
    return best_model

# --- Testing ---
def CNN_Prediction(X_test, y_test, params):
    model = load_model_file()
    model.compile(optimizer='Adam', loss=get_Obj(params['obj']), metrics=['accuracy'])

    y_classes = model.predict(X_test, batch_size=100)

    if len(y_test.shape) > 1:
        y_predict = np.argmax(y_test, axis=1)
    else:
        y_predict = y_test

    y_actual = np.argmax(y_classes, axis=1)

    fscore, acc, cm = evaluate(y_actual, y_predict)
    print('Test fscore:', fscore)
    print('Test accuracy:', acc)
    print('Confusion matrix:\n', cm)
    return

# --- MAIN FLOW ---
if __name__ == "__main__":
    args = parse_args()
    train_params = {
        'do': float(args.do) if args.do else 0.5,
        'a': float(args.a) if args.a else 0.3,
        'k': int(args.k) if args.k else 4,
        'cl': int(args.cl) if args.cl else 5,
        's': float(args.s) if args.s else 1,
        'pf': float(args.pf) if args.pf else 1,
        'pt': args.pt if args.pt else 'Avg',
        'fp': args.fp if args.fp else 'proportional',
        'opt': args.opt if args.opt else 'Adam',
        'obj': args.obj if args.obj else 'ce',
        'patience': int(args.pat) if args.pat else 200,
        'tolerance': float(args.tol) if args.tol else 1.005,
        'res_alias': args.csv if args.csv else 'res' + today,
        'modeltype': args.modeltype if args.modeltype else 'custom'
    }

    (X_train, y_train), (X_val, y_val) = load_data()
    X_train = X_train.reshape(-1, 32, 32, 1)
    X_val = X_val.reshape(-1, 32, 32, 1)

    classes = np.unique(np.argmax(y_train, axis=1))
    y_train_flat = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_flat)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    model = CNN_Train(X_train, y_train, X_val, y_val, train_params, class_weights_dict)
    store_model(model)

    (X_test, y_test) = load_testdata()
    X_test = X_test.transpose(0, 2, 3, 1)
    CNN_Prediction(X_test, y_test, train_params)
