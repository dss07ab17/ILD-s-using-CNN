import re
import os
import sys
import cv2
import pickle
import datetime
import numpy as np
from tensorflow import keras
import argparse
import keras
from keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
import sklearn.metrics as metrics
from keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.initializers import Orthogonal,HeUniform
from keras.models import model_from_json
import matplotlib.pyplot as plt

# Define the target class names
target_names = ['consolidation', 'fibrosis', 'ground_glass', 'healthy', 'micronodules', 'reticulation']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./output/confusion_matrix.png')
    plt.close()

t = datetime.datetime.now()
today = str('_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+':'+str(t.minute))

# debug
from ipdb import set_trace as bp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-do',  help='Dropout param [default: 0.5]')
    parser.add_argument('-a',   help='Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU] [default: 0.3]')
    parser.add_argument('-k',   help='Feature maps k multiplier [default: 4]')
    parser.add_argument('-cl',  help='Number of Convolutional Layers [default: 5]')
    parser.add_argument('-s',   help='Input Image rescale factor [default: 1]')
    parser.add_argument('-pf',  help='Percentage of the pooling layer: [0,1] [default: 1]')
    parser.add_argument('-pt',  help='Pooling type: \'Avg\', \'Max\' [default: Avg]')
    parser.add_argument('-fp',  help='Feature maps policy: \'proportional\',\'static\' [default: proportional]')
    parser.add_argument('-opt', help='Optimizer: \'SGD\',\'Adagrad\',\'Adam\' [default: Adam]')
    parser.add_argument('-obj', help='Minimization Objective: \'mse\',\'ce\' [default: ce]')
    parser.add_argument('-pat', help='Patience parameter for early stoping [default: 200]')
    parser.add_argument('-tol', help='Tolerance parameter for early stoping [default: 1.005]')
    parser.add_argument('-csv', help='csv results filename alias [default: res]')
    args = parser.parse_args()

    return args

def load_data():
    # load the dataset as X_train and as a copy the X_val
    X_train = pickle.load( open( "./pickle/X_train.pkl", "rb" ) ,encoding="latin1")
    y_train = pickle.load( open( "./pickle/y_train.pkl", "rb" ) ,encoding="latin1")
    X_val = pickle.load( open( "./pickle/X_val.pkl", "rb" ) ,encoding="latin1")
    y_val = pickle.load( open( "./pickle/y_val.pkl", "rb" ),encoding="latin1")

    

    # adding a singleton dimension and rescale to [0,1]
    X_train = np.asarray(np.expand_dims(X_train,1))/float(255)
    X_val = np.asarray(np.expand_dims(X_val,1))/float(255)

    # labels to categorical vectors
    uniquelbls = np.unique(y_train)
    nb_classes = uniquelbls.shape[0]
    zbn = np.min(uniquelbls) # zero based numbering
    y_train = to_categorical(y_train-zbn, nb_classes)
    y_val = to_categorical(y_val-zbn, nb_classes)

    return (X_train, y_train), (X_val, y_val)

def load_testdata():

    # load the dataset as X_train and as a copy the X_val
    X_test = pickle.load( open( "./pickle/X_test.pkl", "rb" ) ,encoding="latin1")
    y_test = pickle.load( open( "./pickle/y_test.pkl", "rb" ),encoding="latin1" )
   

    # adding a singleton dimension and rescale to [0,1]
    X_test = np.asarray(np.expand_dims(X_test,1))/float(255)

    # labels to categorical vectors
    # uniquelbls = np.unique(y_test)
    # nb_classes = uniquelbls.shape[0]
    # zbn = np.min(uniquelbls) # zero based numbering
    # only used to make fscore,cm, acc calculation, single dimension required
    # y_test = np_utils.to_categorical(y_test - zbn, nb_classes)
    
    return (X_test, y_test)

def evaluate(actual, pred):
    f_score = metrics.f1_score(actual, pred, average='weighted')
    accuracy = metrics.accuracy_score(actual, pred)
    cm = metrics.confusion_matrix(actual, pred)
    
    # Print classification report with target names
    print("\nClassification Report:")
    print(metrics.classification_report(actual, pred, target_names=target_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    return f_score, accuracy, cm

def store_model(model):
    json_string = model.to_json()
    open('./pickle/ILD_CNN_model.json', 'w').write(json_string)
    model.save_weights('./pickle/ILD_CNN_model_weights.weights.h5')

    return json_string

def load_model():
    model = model_from_json(open('./pickle/ILD_CNN_model.json').read())
    model.load_weights('./pickle/ILD_CNN_model_weights.weights.h5')

    return model


def get_FeatureMaps(L, policy, constant=17):
    return {
        'proportional': (L+1)**2,
        'static': constant,
    }[policy]

def get_Obj(obj):
    return {
        'mse': 'MSE',
        'ce': 'categorical_crossentropy',
    }[obj]

def get_model(input_shape, output_shape, params):

    print('compiling model...')
        
    # Dimension of The last Convolutional Feature Map (eg. if input 32x32 and there are 5 conv layers 2x2 fm_size = 27)
    fm_size = input_shape[-1] - params['cl']
    print ('fm_size : ', fm_size)
    print ('input_shape[-1] : ', input_shape[-1])
    print ('number of convolutional layers : ', params['cl'])
    
    # Tuple with the pooling size for the last convolutional layer using the params['pf']
    pool_siz = (np.round(fm_size*params['pf']).astype(int), np.round(fm_size*params['pf']).astype(int))
    
    # Initialization of the model
    model = Sequential()
    params['fp'] = 'static'  # or 'proportional' based on your model's requirement
    # Add convolutional layers to model
    # model.add(Convolution2D(params['k']*get_FeatureMaps(1, params['fp']), 2, 2, init='orthogonal', activation=LeakyReLU(params['a']), input_shape=input_shape[1:]))
    # added by me
    model.add(Convolution2D(params['k']*get_FeatureMaps(1, params['fp']), 2, 2, kernel_initializer=Orthogonal(),padding="SAME", input_shape=input_shape[1:]))
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    print ('Layer 1 parameters settings:')
    print ('number of filters to be used : ', params['k']*get_FeatureMaps(1, params['fp']))
    print ('kernel size : 2 x 2' )
    print ('input_shape of tensor is : ', input_shape[1:])

    for i in range(2, params['cl']+1):
        # model.add(Convolution2D(params['k']*get_FeatureMaps(i, params['fp']), 2, 2, init='orthogonal', activation=LeakyReLU(params['a'])))
        model.add(Convolution2D(params['k']*get_FeatureMaps(i, params['fp']), 2, 2, kernel_initializer=Orthogonal()))
        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=params['a']))
        print ('Layer',  i, ' parameters settings:')
        print ('number of filters to be used : ', params['k']*get_FeatureMaps(i, params['fp']))
        print ('kernel size : 2 x 2' )


    # Add Pooling and Flatten layers to model
    print ('entering 2D Pooling layer')
    print ('Pooling : ', params['pt'])
    print ('pool_size : ', pool_siz)
    feature_maps = get_FeatureMaps(params['cl'], params['fp'])

    print(f"DEBUG: Input to pooling layer has shape {model.output_shape}")
    input_shape = model.output_shape[1:3]  # Extract (height, width)
    pool_siz = min(input_shape[0], input_shape[1], max(1, int(get_FeatureMaps(params['cl'], params['fp']) / params['pf'])))

   
    if pool_siz > input_shape[0] or pool_siz > input_shape[1]:
        print(f"WARNING: Pool size {pool_siz} is larger than input shape {input_shape}. Adjusting...")
        pool_siz = min(input_shape[0], input_shape[1])  # Adjust to valid size

    print(f"Using pool size: {pool_siz}")
    model.add(AveragePooling2D(pool_size=(pool_siz, pool_siz)))



    model.add(Flatten())

    # dropout is 50% or do=0.5
    model.add(Dropout(params['do']))

    # Add Dense layers and Output to model
    # model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6, init='he_uniform', activation=LeakyReLU(0)))
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp'])/params['pf']*6), kernel_initializer=HeUniform()))
    
    print ('output_dimension : ', int(params['k']*get_FeatureMaps(params['cl'], params['fp'])/params['pf']*6))

    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(params['do']))

    
    # model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*2, init='he_uniform', activation=LeakyReLU(0)))
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp'])/params['pf']*2), kernel_initializer=HeUniform()))
    #  model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(params['do']))
    model.add(Dense(output_shape[1], kernel_initializer=HeUniform(), activation='softmax'))



#sk modif for decay, works only with ADAM

    decay = 0.001
    lr=0.001
    
    # Compile model and select optimizer and objective function
    if params['opt'] not in ['Adam', 'Adagrad', 'SGD']:
        sys.exit('Wrong optimizer: Please select one of the following. Adam, Adagrad, SGD')
    if get_Obj(params['obj']) not in ['MSE', 'categorical_crossentropy']:
        sys.exit('Wrong Objective: Please select one of the following. MSE, categorical_crossentropy')
#    model.compile(optimizer=params['opt'], loss=get_Obj(params['obj']))

    optimizer=keras.optimizers.Adam(learning_rate=lr,decay=decay)
    model.compile(optimizer=optimizer, loss=get_Obj(params['obj']))

    return model

def CNN_Train(x_train, y_train, x_val, y_val, params,class_weights):
    ''' TODO: documentation '''

    
    # Parameters String used for saving the files
    parameters_str = str('_d' + str(params['do']).replace('.', '') +
                         '_a' + str(params['a']).replace('.', '') + 
                         '_k' + str(params['k']).replace('.', '') + 
                         '_c' + str(params['cl']).replace('.', '') + 
                         '_s' + str(params['s']).replace('.', '') + 
                         '_pf' + str(params['pf']).replace('.', '') + 
                         '_pt' + params['pt'] +
                         '_fp' + str(params['fp']).replace('.', '') +
                         '_opt' + params['opt'] +
                         '_obj' + params['obj'])

    # Printing the parameters of the model
    print('[Dropout Param] \t->\t'+str(params['do']))
    print('[Alpha Param] \t\t->\t'+str(params['a']))
    print('[Multiplier] \t\t->\t'+str(params['k']))
    print('[Patience] \t\t->\t'+str(params['patience']))
    print('[Tolerance] \t\t->\t'+str(params['tolerance']))
    print('[Input Scale Factor] \t->\t'+str(params['s']))
    print('[Pooling Type] \t\t->\t'+ params['pt'])
    print('[Pooling Factor] \t->\t'+str(str(params['pf']*100)+'%'))
    print('[Feature Maps Policy] \t->\t'+ params['fp'])
    print('[Optimizer] \t\t->\t'+ params['opt'])
    print('[Objective] \t\t->\t'+ get_Obj(params['obj']))
    print('[Results filename] \t->\t'+str(params['res_alias']+parameters_str+'.txt'))

    # Rescale Input Images
    if params['s'] != 1:
        print('\033[93m'+'Rescaling Patches...'+'\033[0m')
        x_train = np.asarray(np.expand_dims([cv2.resize(x_train[i, 0, :, :], (0,0), fx=params['s'], fy=params['s']) for i in xrange(x_train.shape[0])], 1))
        x_val = np.asarray(np.expand_dims([cv2.resize(x_val[i, 0, :, :], (0,0), fx=params['s'], fy=params['s']) for i in xrange(x_val.shape[0])], 1))
        print('\033[92m'+'Done, Rescaling Patches'+'\033[0m')
        print('[New Data Shape]\t->\tX: '+str(x_train.shape))

    print ('x_shape is: ', x_train.shape)

    if os.path.exists('./pickle/ILD_CNN_model.h5'):
        print('Model exists')  
        model = load_model('./pickle/ILD_CNN_model.h5')
    else:
        print ('restart from 0')
        model = get_model(x_train.shape, y_train.shape, params)
    # Counters-buffers
    maxf         = 0
    maxacc       = 0
    maxit        = 0
    maxtrainloss = 0
    maxvaloss    = np.inf
    best_model   = model
    it           = 0    
    p            = 0
    # Open file to write the results
    

    # Remove invalid characters for filenames
    safe_res_alias = re.sub(r'[\\/*?:"<>|]', '_', params['res_alias'])
    safe_parameters_str = re.sub(r'[\\/*?:"<>|]', '_', parameters_str)

    open('./output/' + safe_res_alias + safe_parameters_str + '.csv', 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')
    open('./output/' + safe_res_alias + safe_parameters_str + '-Best.csv', 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')


    print ('starting the loop of training with number of patience = ', params['patience'])
    
    while p < params['patience']:
        p += 1

        # Fit the model for one epoch
        print('Epoch: ' + str(it))
        history = model.fit(x_train, y_train, batch_size=250, epochs=1, validation_data=(x_val,y_val), shuffle=True,class_weight=class_weights)

    
        # Evaluate models
        y_score = model.predict(x_val, batch_size=1050)

        fscore, acc, cm = evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))
        print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))

        # Write results in file

        # Fix the filename by removing forbidden characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', params['res_alias'] + parameters_str)  

        # Ensure the directory exists before writing the file
        
        os.makedirs('./output/', exist_ok=True)

        # Write results in a properly formatted file
        with open(f'./output/{safe_filename}.csv', 'a') as file:
           file.write(f"{it}, {fscore}, {acc}, {np.max(history.history['loss'])}, {np.max(history.history['val_loss'])}\n")


        # check if current state of the model is the best and write evaluation metrics to file
        if fscore > maxf*params['tolerance']:  # if fscore > maxf*params['tolerance']:
            print ('fscore is still bigger than last iterations fscore + 5%')
            #p            = 0  # restore patience counter
            best_model   = model  # store current model state
            maxf         = fscore 
            maxacc       = acc
            maxit        = it
            maxtrainloss = np.max(history.history['loss'])
            maxvaloss    = np.max(history.history['val_loss'])

            print(np.round(100*cm/np.sum(cm,axis=1).astype(float)))

            # ✅ Ensure the output directory exists
            # ✅ Ensure the output directory exists
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', params['res_alias'] + parameters_str)  

            # Ensure the directory exists before writing the file
        
            os.makedirs('./pickle/', exist_ok=True)

            # Write results in a properly formatted file
            with open(f'./pickle/{safe_filename}.csv', 'a') as file:
                file.write(f"{it}, {fscore}, {acc}, {np.max(history.history['loss'])}, {np.max(history.history['val_loss'])}\n")
            store_model(best_model)
        it += 1
    
    print('Max: fscore:', maxf, 'acc:', maxacc, 'epoch: ', maxit, 'train loss: ', maxtrainloss, 'validation loss: ', maxvaloss)

    return best_model



def CNN_Prediction(X_test, y_test, params):
    f=open ('./output/res.txt','w')
    model = load_model()
    model.compile(optimizer='Adam', loss=get_Obj(params['obj']))

    y_classes = model.predict(X_test, batch_size=100)
    y_val_subset = y_classes[:]
    y_test_subset = y_test[:]  # Check its shape

    # Fix: Only apply argmax if y_test is one-hot encoded
    if len(y_test_subset.shape) > 1:  
        y_predict = np.argmax(y_test_subset, axis=1)  
    else:  
        y_predict = y_test_subset  # Already class labels

    y_actual = np.argmax(y_val_subset, axis=1)  # Convert model output to class labels

    fscore, acc, cm = evaluate(y_actual, y_predict)



    print ('f-score is : ', fscore)
    print ('accuracy is : ', acc)
    print ('confusion matrix')
    print (cm)
    f.write('f-score is : '+ str(fscore)+'\n')
    f.write( 'accuracy is : '+ str(acc)+'\n')
    f.write('confusion matrix\n')
    n= cm.shape[0]
    for i in range (0,n):
        for j in range (0,n):
           f.write(str(cm[i][j])+' ')
        f.write('\n')
    f.close()
    open('./' + 'TestLog.csv', 'a').write(str(params['res_alias']) + ', ' + str(str(fscore) + ', ' + str(acc)+'\n'))
    return

# initialization
args         = parse_args()                          # Function for parsing command-line arguments
train_params = {
     'do' : float(args.do) if args.do else 0.5,        # Dropout Parameter
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 200,       # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res' + str(today)     # csv results filename alias
}

# loading patch data
(X_train, y_train), (X_val, y_val) = load_data()
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
X_train = X_train.reshape(-1, 32, 32, 1)  # Ensure proper input shape
X_val = X_val.reshape(-1, 32, 32, 1)
classes = np.unique(y_train)
y_train_flat = np.ravel(y_train) 
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_flat)
class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}


# train a CNN model
model = CNN_Train(X_train, y_train, X_val, y_val, train_params,class_weights_dict)

# store the model and weights
store_model(model)

print ('training completed')
print ('loading test set')

# load test data set 
(X_test, y_test) = load_testdata()
X_test = X_test.transpose(0, 2, 3, 1)

# predict with test dataset and record results
pred = CNN_Prediction(X_test, y_test, train_params)

print ('assessment with test set completed')