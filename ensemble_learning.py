'''
This is the code base used for the following: 
(i) UNet based semantic semgnetation to create lung masks for the datasets 
used in this study; (ii) perform repeated CXR-specific pretraining; (ii) 
Fine-tuning on COVID-19 detection; (iv) create ensembles 
of fine-tuned models to improve performance; 
'''
#%% load libraries; clear current gpu session
import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) #because of numpy version
import tensorflow as tf
from keras import backend as K
K.clear_session()

#%%
from keras.models import Model, Input
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, SeparableConv2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D,Flatten,Average, Dropout
#import time
#from lime import lime_image
from skimage.segmentation import mark_boundaries
import statistics
from keras import applications
from scipy import interp
import cv2
import imutils
import pickle
import struct
import shutil
import numpy as np
import zlib
import argparse
from math import pi
from math import cos
from math import floor
from keras import backend
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score, log_loss
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import scikitplot as skplt
from itertools import cycle
from sklearn.utils import class_weight
from keras.models import load_model, Model, Sequential, Input
import numpy as np
import itertools
from keras.utils import plot_model, to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from scipy import interp
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.ndimage.interpolation import zoom
import numpy as np
from keras.backend import tensorflow_backend
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import statistics
from sklearn import metrics
from scipy.optimize import minimize
from classification_models.keras import Classifiers
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score, log_loss
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Concatenate, SeparableConv2D
from keras.layers import SeparableConv2D
from scipy.ndimage.interpolation import zoom
import statistics 
from skimage.segmentation import mark_boundaries
import numpy as np
from keras.backend import tensorflow_backend
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D
import time
from scipy import interp
import cv2
import imutils
import pickle
import struct
import shutil
import numpy as np
import zlib
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam, SGD
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from keras.callbacks import CSVLogger
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
import itertools
from itertools import cycle
from sklearn.utils import class_weight
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pydicom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import random
from shutil import copyfile
from tensorflow.python.framework import ops
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import backend as K
from keras import applications
import os, argparse
import pickle
import cv2, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
import joblib
from keras import backend as K
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import keras
from classification_models.keras import Classifiers
import efficientnet.keras as efn 
import math
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Conv2D, Dense, Concatenate, Flatten, MaxPooling2D, SeparableConv2D, Activation, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D,Flatten,Average, BatchNormalization, Dropout
import time
import keras
import cv2
import imutils
from sklearn.metrics import fbeta_score, f1_score, cohen_kappa_score, precision_score, recall_score
import pandas as pd
import numpy as np
from scipy import interp
from keras.layers import concatenate
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import pickle
from keras.optimizers import Adam, SGD
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score
from sklearn.metrics import average_precision_score
from keras.callbacks import CSVLogger
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import scikitplot as skplt
from itertools import cycle
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import itertools
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import backend as K
from kerassurgeon.identify import get_apoz
from kerassurgeon import Surgeon
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
from matplotlib import pyplot as plt
from glob import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
#%% 
#get current working directory
os.getcwd()

#%% 
# define custom function for confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
# define custom function to convert DICOM and jpeg images to png, if any:

def dcm2png(input_dir: str, output_dir: str):
    """
    Becareful all output images are gray image with 8 bit
    :param input_dir: dcm file directory
    :param output_dir: save directory
    """
    if not os.path.isdir(input_dir):
        raise ValueError("Input dir is not found!")

    if not os.path.isdir(output_dir):
        raise ValueError("Out dir is not found!")

    img_list = [f for f in os.listdir(input_dir)
                if f.split('.')[-1] == 'dcm' or f.split('.')[-1] == 'jpeg'] 
    for n, f in enumerate(img_list):
        
        if f.split(".")[-1] == "dcm":
            dcm_file = input_dir + f
            ds = pydicom.dcmread(dcm_file)
            pixel_array_numpy = ds.pixel_array
            pixel_array_numpy = cv2.normalize(pixel_array_numpy,
                                              None,
                                              alpha=0,
                                              beta=255,
                                              norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_8UC1)
            pixel_array_numpy = cv2.resize(pixel_array_numpy,
                                           (224,224))
            img_file = output_dir + f.replace('.dcm', '.png')
            cv2.imwrite(img_file, pixel_array_numpy)

        else:
            
            if f.split(".")[-1] == "jpeg":
                image_file = input_dir + f
                pixel_array_numpy = cv2.imread(image_file)
                pixel_array_numpy = cv2.cvtColor(pixel_array_numpy, cv2.COLOR_BGR2GRAY)
                pixel_array_numpy = cv2.resize(pixel_array_numpy,
                                               (224,224))
                image_file = output_dir + f.replace('.jpeg', '.png')
                cv2.imwrite(image_file, pixel_array_numpy)
        
                
                if n % 50 == 0:
                    print('{} image processed'.format(n))

#usage
#dcm2png("../source_folder/", "../destination_folder/")

#%% 
# Begin U-Net based semantic segmentation to genrate lung masks for the input CXRs
#define different loss functions

def dice_loss(y_true, y_pred):
    num = 2 * np.sum(np.multiply(y_true, y_pred))
    den = np.sum(y_true) + np.sum(y_pred)
    return num/den

def bce_dice_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = np.sum(np.multiply(y_true, y_pred))
    union = np.sum(y_true+y_pred) - intersection
    return intersection/union  

#%%
'''
data organization: data
                    |>train
                        |>image (original images)
                        |>label (training masks)
                    |>val
                        |>image (original images)
                        |>label (validation masks)
                    |>test (all test images)
                    |>result (will contain the generated masks)
'''
#%%
# Define the UNET model architecture with an empirically determined dropout ratio of 0.5.
# The model operates with 256 by 256 inputs, 
def unetdrop(pretrained_weights = None,input_size = (256,256,1)): 
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Add()([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Add()([conv3,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Add()([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Add()([conv1,up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = [bce_dice_loss], metrics = [iou, 'accuracy'])   
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#%%
# Define the data format and the functions to train and test 
#with the data using image generators. Make sure to use the same seed for 
# image_datagen and mask_datagen to ensure the transformation for image 
# and mask is the same. If you want to visualize the 
# results of generator, set save_to_dir = "your path"

#create a color dictionary
A = [128,128,128]
B = [128,0,0]
C = [192,192,128]
D = [128,64,128]
E = [60,40,222]
F = [128,128,0]
G = [192,128,128]
H = [64,64,128]
I = [64,0,128]
J = [64,64,0]
K = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([A, B, C, D, E, F, G, H, I, J, K, Unlabelled])

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],
                                        new_mask.shape[1]*new_mask.shape[2],
                                        new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
        
def valGenerator(batch_size,val_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path,target_size = (256,256),flag_multi_class = False,as_gray = True): 
    for filename in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,filename),as_gray = as_gray) 
        img = img / 255.
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,test_path, flag_multi_class = False,num_class = 2):
    file_names = os.listdir(test_path)
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,file_names[i]),img)

#%% 
#train model with data augmentations
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'C:/Users/data/train','image','label',
                        data_gen_args,save_to_dir = None) #batch size = 2 here
valGene = valGenerator(2,'C:/Users/data/val','image','label',
                       data_gen_args,save_to_dir = None) #batch size = 2 here

model = unetdrop()
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                               epsilon=1e-4, mode='min'),
             ModelCheckpoint(monitor='val_loss', filepath='unetdrop.hdf5', save_best_only=True,
                             mode='min', verbose = 1)]

model.fit_generator(generator=myGene,steps_per_epoch=273, epochs=200, callbacks=callbacks,
                    validation_data=valGene, validation_steps=77, verbose=1)

#steps_per_epoch = no. of training samples//batch_size, add 1 if not absolutely divisible
#validation_steps = no. of validation samples//batch_size, add 1 if not absolutely divisible

#%%
#model validation
test_path = "../test/'
save_path = "../result/"
data_gen_args = dict(rotation_range=10.,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=5,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest') 

testGene = testGenerator(test_path)
model = unetdrop()
model.load_weights("../unetdrop.hdf5")
results = model.predict_generator(testGene,72,verbose=1, workers=1, use_multiprocessing=False) 
#steps per epoch is the no. of samples in test image.
saveResult(save_path, results, test_path)

#%%
# postprocessing with the mask and image: 
#This script helps to postprocess the images with the mask generated through 
#the UNET and relax the boundaries by 5% on top, bottom, left, and right, and store 
#the bounding box cordinates to a csv file. The cropped bounding box images are stored to a directory.
# the original GT disease annotations, if available, will also be rescaled and stored to
# a separate CSV file.
#the source CSV file in organized as follows:
#filename, x, y, width, height, target, class: x, y denote the top left
#target: 1 or 0 based on your labels; class: COVID-19

def generate_bounding_box(image_dir: str,
                          mask_dir: str,
                          orgin_csv: str,
                          dest_csv: str,
                          crop_save_dir: str,
                          show_results: bool=False):
    if not os.path.isdir(mask_dir):
        raise ValueError("mask_dir not existed")

    csv_file_data = []
    with open(orgin_csv, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            csv_file_data.append(row)

    with open(dest_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        for j, row in enumerate(csv_file_data[1:]):
            case_name = row[0] + '.png' #all mask files are png file type            
            mask = cv2.imread(mask_dir + case_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = cv2.imread(image_dir + case_name, cv2.COLOR_BGR2GRAY)
            #original images are resized to 256x256, 
            #comment this if you want to keep the original image resolution
            image = cv2.resize(image,(256,256)) 
            if mask is None or image is None:
                #raise ValueError("The image can not be read: " + case_name)
                cv2.imwrite(crop_save_dir + case_name, image) 
            
            reduce_col = np.sum(mask, axis=1)
            reduce_row = np.sum(mask, axis=0)
            # many 0s add up to none zero, we need to threshold it
            reduce_col = (reduce_col >= 255)*reduce_col
            reduce_row = (reduce_row >= 255)*reduce_row
            first_none_zero = None
            last_none_zero = None

            last = 0
            for i in range(reduce_col.shape[0]):
                current = reduce_col[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_col[i]

            up = first_none_zero
            down = last_none_zero

            first_none_zero = None
            last_none_zero = None
            last = 0
            for i in range(reduce_row.shape[0]):
                current = reduce_row[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_row[i]

            left = first_none_zero
            right = last_none_zero
            
            if up is None or down is None or left is None or right is None:
                raise ValueError("The border is not found: " + case_name)
            
            # new coordinates for image which is 1 times of mask, mask images are 256x256, 
            #need to multiply 1 times to get 256x256, and relaxing the borders by 5% on all directions
            # for example, if the original image resolution is 
            #1024x1024, new coordinates for image is 4 times of mask (256x256), 
            #need to multiply 4 times to 1024x1024, and relaxing the borders by 5% on all directions, 
            #i.e. int(4 * (down - up + 1) * 0.05)
            
            loose = int(1 * (down - up + 1) * 0.05) 
            
            image_up = 1 * up - loose #4
            if image_up < 0:
                image_up = 0
            image_down = 1*(down+1)+loose #4
            if image_down > image.shape[0] + 1:
                image_down = image.shape[0]

            loose2 = int(1 * (right - left + 1) * 0.05) #4
            image_left = 1 * left - loose2 #4
            if image_left < 0:
                image_left = 0
            image_right = 1*(right+1)+loose2 #4
            if image_right > image.shape[1] + 1:
                image_right = image.shape[1]

            crop = image[image_up: image_down, image_left: image_right]
            crop = cv2.resize(crop, (256,256)) #1024, 1024 
            cv2.imwrite(crop_save_dir + case_name, crop)

            # write to new csv
            crop_width = image_right - image_left + 1
            crop_height = image_down - image_up + 1

            if row[6] == "COVID-19":

                y_scale_change = 256 / crop_height #1024 
                x_scale_change = 256 / crop_width # 1024 
                bbox_y = int(float(row[2])) 
                #new_y = int((bbox_y - image_up) * y_scale_change)
                new_y = int((bbox_y/4 - image_up) * y_scale_change) #since resized to 256 from 1024, scale by 1/4

                bbox_x = int(float(row[1]))
                #new_x = int((bbox_x - image_left) * x_scale_change)
                new_x = int((bbox_x/4 - image_left) * x_scale_change)  #since resized to 256 from 1024, scale by 1/4

                bbox_width = int(float(row[3]))
                bbox_height = int(float(row[4]))
                new_width = int(bbox_width/4 * x_scale_change)  #since resized to 256 from 1024, scale by 1/4
                new_height = int(bbox_height/4 * y_scale_change) #since resized to 256 from 1024, scale by 1/4

                csv_writer.writerow([case_name,
                                     image_left,
                                     image_up,
                                     crop_width,
                                     crop_height,
                                     new_x,
                                     new_y,
                                     new_width,
                                     new_height,
                                     row[6]])

            else:
                csv_writer.writerow([case_name,
                                     image_left,
                                     image_up,
                                     crop_width,
                                     crop_height,
                                     "",
                                     "",
                                     "",
                                     "",
                                     row[6]])
            
            if j % 100 == 0:
                print(j, " images are processed!")

#usage:
generate_bounding_box("../original_data/",
                      "../mask/",
                      '../ground truth annotations.csv',
                      '../destination_annotations.csv',
                      "../bounding_box_crop/",
                      show_results=False)

#%%
'''
Now that we have preprocessed the orignal CXRs and cropped them to 
a size of the bounding box with 256x256 grayscale pixel resolution, 
we can begin training the models. 
The first step in repeated CXR_specific pretrainig is to train the 
models on a large scale selection of CXRs to clasify them into normal and abnormal classes.
'''
#%% Loading the training data
#We split the data at the patient-level into 90% for training and 10% for testing
#10% of the training data is randomly allocated for validation
img_width, img_height = 256,256
train_data_dir = "../first_level_pretraining/train"
test_data_dir = "../first_level_pretraining/test"
epochs = 16
batch_size = 16
num_classes = 2 #abnormal, normal
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
#define data generators
datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) #90/10, no augmentation except rescaling

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'training')

validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'validation')

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle = False)

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#true labels
Y_test=test_generator.classes
print(Y_test.shape)

#convert test labels to categorical
Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)

#%%
#compute class weights to penalize over represented classes

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
print(class_weights)
#%%
'''
Declare model architectures: The models used in this study are:
a custom WRN, VGG-16, VGG-19, Inception-V3, Xception, NasNet-Mobile, 
DenseNet-121, MobileNet-V2, and ResNet-18.
The custom D-WRN has a depth of 16 and width of 4, and dropout = 0.3. 
Compute N = (n - 4) / 6.
Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
param k: Width of the network.
param dropout: Adds dropout if value is greater than 0.0
param verbose: Debug info to describe created WRN

Example:
model_wrn = create_wide_residual_network(ip, nb_classes=2, N=2, k=4, dropout=0.3, verbose=1)

model = Model(ip, model_wrn)
'''
#%%
#declare the architecture for the custom WRN used in this study
weight_decay = 0.0005

def initial_conv(input):
    x = Conv2D(16, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x

def expand_conv(init, base, k, strides=(1, 1)):
    x = Conv2D(base * k, (5, 5), padding='same', strides=strides, kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Conv2D(base * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    skip = Conv2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(init)

    m = Add()([x, skip])

    return m

def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (5, 5), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, 
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, W_regularizer=l2(weight_decay), 
              activation='softmax')(x)

    model = Model(ip, x, name = 'wide-residual-cnn')

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

model_wrn = create_wide_residual_network(input_shape, nb_classes=2, 
                                                       N=2, k=4, dropout=0.3, verbose=1) 
model_wrn.summary()

#%%
# declare the architecture of the customized pretrained models for this study.
'''
The models, for the first level of pretraining, are truncated at 
these empirically determined intermediate layers and are appended with 
(i) zero-padding, (i) a 3 x 3 convolutional layer with 1024 feature maps, 
(ii) a global average pooling (GAP) layer, 
(iii) a dropout layer with an empirically determined dropout ratio of 0.5, 
and (iv) a final dense layer with Softmax activation 
to output prediction probabilities. 
These customized ImageNet-pretrained models are retrained to learn CXR 
domain-specific feature representations 
to classify CXRs as belonging to normal or abnormal classes. 

Model	Truncated layers
VGG-16	Block5-conv3
VGG-19	Block5-conv4
Inception-V3	Mixed3
Xception	Add_3
DenseNet-121	Pool3_pool
MobileNet-V2	Block_9_add
NASNet-mobile	Activation_94
ResNet-18	Add_6
'''
#%%
#begin model definitions
#forked from https://github.com/qubvel/classification_models
ResNet18, preprocess_input = Classifiers.get('resnet18')
# build model
resnet18_cnn = ResNet18(input_tensor=model_input, 
                        weights='imagenet', include_top=False)
resnet18_cnn.summary()
#%%
base_model_resnet18=Model(inputs=resnet18_cnn.input,
                        outputs=resnet18_cnn.get_layer('add_6').output)
x = base_model_resnet18.output
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_resnet18')(x)
x = GlobalAveragePooling2D()(x)              
x = Dropout(0.5)(x)                     
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_resnet18 = Model(inputs=base_model_resnet18.input, 
                    outputs=predictions, 
                    name = 'resnet18_new_first_pretrained')
model_resnet18.summary()

#%%
xception_cnn = Xception(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
xception_cnn.summary()

#%%
base_model_xception=Model(inputs=xception_cnn.input,
                        outputs=xception_cnn.get_layer('add_3').output)
x = base_model_xception.output
x = Activation('relu')(x)       
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_xception')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_xception = Model(inputs=base_model_xception.input, 
                    outputs=predictions, 
                    name = 'xception_new_first_pretrained')
model_xception.summary()

#%%
vgg16_cnn = VGG16(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
vgg16_cnn.summary()

#%%
base_model_vgg16=Model(inputs=vgg16_cnn.input,
                        outputs=vgg16_cnn.get_layer('block5_conv3').output)
x = base_model_vgg16.output    
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_vgg16')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg16 = Model(inputs=base_model_vgg16.input, 
                    outputs=predictions, 
                    name = 'vgg16_new_first_pretrained')
model_vgg16.summary()

#%%
vgg19_cnn = VGG19(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
vgg19_cnn.summary()

#%%
base_model_vgg19=Model(inputs=vgg19_cnn.input,
                        outputs=vgg19_cnn.get_layer('block5_conv4').output)
x = base_model_vgg19.output    
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_vgg19')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg19 = Model(inputs=base_model_vgg19.input, 
                    outputs=predictions, 
                    name = 'vgg19_new_first_pretrained')
model_vgg19.summary() 

#%%    
iv3_cnn = InceptionV3(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
iv3_cnn.summary()

#%%
base_model_iv3=Model(inputs=iv3_cnn.input,
                        outputs=iv3_cnn.get_layer('mixed3').output)
x = base_model_iv3.output    
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_iv3')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_iv3 = Model(inputs=base_model_iv3.input, 
                    outputs=predictions, 
                    name = 'iv3_new_first_pretrained')
model_iv3.summary() 

#%%
densenet_cnn = DenseNet121(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
densenet_cnn.summary()

#%%
base_model_densenet=Model(inputs=densenet_cnn.input,
                        outputs=densenet_cnn.get_layer('pool3_pool').output)
x = base_model_densenet.output    
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_densenet')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_densenet = Model(inputs=base_model_densenet.input, 
                    outputs=predictions, 
                    name = 'densenet_new_first_pretrained')
model_densenet.summary() 

#%%
mobilev2_cnn = MobileNetV2(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
mobilev2_cnn.summary()

#%%
base_model_mobilev2=Model(inputs=mobilev2_cnn.input,
                        outputs=mobilev2_cnn.get_layer('block_9_add').output)

x = base_model_mobilev2.output
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_mobilev2')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_mobilev2 = Model(inputs=base_model_mobilev2.input, 
                    outputs=predictions, 
                    name = 'mobilev2_new_first_pretrained')
model_mobilev2.summary()

#%%
nasnet_cnn = NASNetMobile(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
nasnet_cnn.summary()

#%%
base_model_nasnet=Model(inputs=nasnet_cnn.input,
                        outputs=nasnet_cnn.get_layer('activation_94').output)
x = base_model_nasnet.output
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(1024, (3, 3), activation='relu', name='extra_conv_nasnet')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
model_nasnet = Model(inputs=base_model_nasnet.input, 
                    outputs=predictions, 
                    name = 'nasnet_new_first_pretrained')
model_nasnet.summary()

#%%
''' now that we have declared all the model architecture, begin training the models
on the first level CXR data
'''
#%% compile and train the models
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_vgg16.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_vgg16.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 
#see if the steps per epochs and validation steps are absolutely divisble, otherwise add 1
 
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg19.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_vgg19.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_vgg19.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_iv3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_iv3.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_iv3.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_xception.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_xception.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_xception.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_densenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_densenet.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_densenet.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_nasnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_nasnet.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_nasnet.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 


#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_resnet18.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_resnet18.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_resnet18.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_mobilev2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_mobilev2.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_mobilev2.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

#%% for the custom wrn model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_wrn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_wrn.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_wrn.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
'''
now that all the models are trained, make predictions with the test data.
here we show it for a single model, repeat for other models
'''
#%%
vgg16_model = load_model('weights/vgg16_new_first_pretrained.06-0.6874.h5')
vgg16_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#%%
#measure performance on test data, first reset the test generator otherwise it gives wierd results
test_generator.reset()

#evaluate accuracy 
custom_y_pred = vgg16_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1) #if not absolutely divisible, otherwise remove 1

#save the predictions
np.savetxt('weights/vgg16_y_pred.csv',custom_y_pred,fmt='%f',delimiter = ",")
np.savetxt('weights/Y_test.csv',Y_test,fmt='%i',delimiter = ",")

#%%
#measure performance
accuracy = accuracy_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The test accuracy of the Custom model is: ', accuracy)

prec = precision_score(Y_test,custom_y_pred.argmax(axis=-1), 
                       average='micro') #options: macro, weighted
print('The precision of the Custom model is: ', prec)

rec = recall_score(Y_test,custom_y_pred.argmax(axis=-1), 
                   average='micro')
print('The recall of the Custom model is: ', rec)

f1 = f1_score(Y_test,custom_y_pred.argmax(axis=-1), 
              average='micro')
print('The f1-score of the Custom model is: ', f1)

mat_coeff = matthews_corrcoef(Y_test,custom_y_pred.argmax(axis=-1))
print('The MCC of the Custom model is: ', mat_coeff)

#Cohen’s kappa: a statistic that measures inter-annotator agreement.
kappa = cohen_kappa_score(Y_test,
                          custom_y_pred.argmax(axis=-1))
print('The cohen kappa score of the Custom model is: ', kappa)

#%% print classification report and plot confusion matrix
import itertools

target_names = ['Abnormal','Normal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix using scikit learn
plt.figure(figsize=(15,10), dpi=300)
plot_confusion_matrix(cnf_matrix, classes=target_names)
plt.show()

#%%
#plot ROC curves

fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test1[:, i], custom_y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test1.ravel(), custom_y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

#%%
# Zoom in view of the upper left corner.
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

#%%
#compute precision-recall curves
colors = cycle(['red', 'blue', 'green', 'cyan', 'teal'])

plt.figure(figsize=(15,10), dpi=300)
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test1[:, i],
                                                        custom_y_pred[:, i])
    average_precision[i] = average_precision_score(Y_test1[:, i], custom_y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test1.ravel(),
   custom_y_pred.ravel())
average_precision["micro"] = average_precision_score(Y_test1, custom_y_pred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(num_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.05)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class PR curve')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()

#%%
'''
repeat the above steps to evaluate the indivdual 
first-level pretrained models and store the weights for use in second-level pretraining.
lets begin second-level pretraining to classify the CXRs into 
normal, bacterial, and non-covid-19 proven viral pneumonia classes
'''
#%%
img_width, img_height = 256,256
train_data_dir = "../second_level_pretraining/train"
test_data_dir = "../second_level_pretraining/test"
epochs = 16
batch_size = 16
num_classes = 3 # normal, bacterial, and viral
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
#define data generators
datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) #90/10, no augmentation except rescaling

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'training')

validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'validation')

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle = False)

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#true labels
Y_test=test_generator.classes
print(Y_test.shape)

#convert test labels to categorical
Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)

#%%
'''
initialize the models from the first level of pretraining 
The models are truncated at their deepest convolutional layer 
and appended with (i) GAP layer, (ii) dropout layer (ratio = 0.5), and 
(iii) dense layer with Softmax activation to output class probabilities
for normal, bacterial and non-covid-19 viral proven CXRs.  
'''
#%%
resnet18_cnn = load_model('weights/resnet18_new_first_pretrained.03-0.6821.h5')
resnet18_cnn.summary()

#%%
base_model_resnet18=Model(inputs=resnet18_cnn.input,
                        outputs=resnet18_cnn.get_layer('extra_conv_resnet18').output)
x = base_model_resnet18.output
x = GlobalAveragePooling2D()(x)              
x = Dropout(0.5)(x)                     
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_resnet18 = Model(inputs=base_model_resnet18.input, 
                    outputs=predictions, 
                    name = 'resnet18_second_pretrained')
model_resnet18.summary()

#%%
xception_cnn = load_model('weights/xception_new_first_pretrained.05-0.6727.h5')
xception_cnn.summary()

#%%
base_model_xception=Model(inputs=xception_cnn.input,
                        outputs=xception_cnn.get_layer('extra_conv_xception').output)
x = base_model_xception.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_xception = Model(inputs=base_model_xception.input, 
                    outputs=predictions, 
                    name = 'xception_second_pretrained')
model_xception.summary()

#%%
vgg16_cnn = load_model('weights/vgg16_new_first_pretrained.06-0.6874.h5')
vgg16_cnn.summary()

#%%
base_model_vgg16=Model(inputs=vgg16_cnn.input,
                        outputs=vgg16_cnn.get_layer('extra_conv_vgg16').output)
x = base_model_vgg16.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg16 = Model(inputs=base_model_vgg16.input, 
                    outputs=predictions, 
                    name = 'vgg16_second_pretrained')
model_vgg16.summary()

#%%
vgg19_cnn = load_model('weights/vgg19_new_first_pretrained.05-0.6913.h5')
vgg19_cnn.summary()

#%%
base_model_vgg19=Model(inputs=vgg19_cnn.input,
                        outputs=vgg19_cnn.get_layer('extra_conv_vgg19').output)
x = base_model_vgg19.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg19 = Model(inputs=base_model_vgg19.input, 
                    outputs=predictions, 
                    name = 'vgg19_second_pretrained')
model_vgg19.summary() 

#%%    
iv3_cnn = load_model('weights/iv3_new_first_pretrained.05-0.6842.h5')
iv3_cnn.summary()

#%%
base_model_iv3=Model(inputs=iv3_cnn.input,
                        outputs=iv3_cnn.get_layer('extra_conv_iv3').output)
x = base_model_iv3.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_iv3 = Model(inputs=base_model_iv3.input, 
                    outputs=predictions, 
                    name = 'iv3_second_pretrained')
model_iv3.summary() 

#%%
densenet_cnn = load_model('weights/densenet_new_first_pretrained.06-0.6827.h5')
densenet_cnn.summary()

#%%
base_model_densenet=Model(inputs=densenet_cnn.input,
                        outputs=densenet_cnn.get_layer('extra_conv_densenet').output)
x = base_model_densenet.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_densenet = Model(inputs=base_model_densenet.input, 
                    outputs=predictions, 
                    name = 'densenet_second_pretrained')
model_densenet.summary() 

#%%
mobilev2_cnn = load_model('weights/mobilev2_new_first_pretrained.06-0.6844.h5')
mobilev2_cnn.summary()

#%%
base_model_mobilev2=Model(inputs=mobilev2_cnn.input,
                        outputs=mobilev2_cnn.get_layer('extra_conv_mobilev2').output)

x = base_model_mobilev2.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_mobilev2 = Model(inputs=base_model_mobilev2.input, 
                    outputs=predictions, 
                    name = 'mobilev2_second_pretrained')
model_mobilev2.summary()

#%%
nasnet_cnn = load_model('weights/nasnet_new_first_pretrained.07-0.6820.h5')
nasnet_cnn.summary()

#%%
base_model_nasnet=Model(inputs=nasnet_cnn.input,
                        outputs=nasnet_cnn.get_layer('extra_conv_nasnet').output)
x = base_model_nasnet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
model_nasnet = Model(inputs=base_model_nasnet.input, 
                    outputs=predictions, 
                    name = 'nasnet_second_pretrained')
model_nasnet.summary()

#%%
#custom WRN:
wrn_cnn = load_model('weights/wrn.07-0.6696.h5')
wrn_cnn.summary()

#%%
base_model_wrn=Model(inputs=wrn_cnn.input,
                        outputs=nasnet_cnn.get_layer('global_average_pooling2d').output)
x = base_model_wrn.output
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
model_wrn = Model(inputs=base_model_wrn.input, 
                    outputs=predictions, 
                    name = 'wrn_second_pretrained')
model_wrn.summary()

#%% 
''' 
after initializing these customized models, we train and validate the models
on the second-level pretraining data
'''
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_vgg16.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_vgg16.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 
#see if the steps per epochs and validation steps are absolutely divisble, otherwise add 1
 
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg19.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_vgg19.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_vgg19.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_iv3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_iv3.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_iv3.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_xception.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_xception.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_xception.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_densenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_densenet.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_densenet.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_nasnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_nasnet.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_nasnet.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 


#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_resnet18.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_resnet18.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_resnet18.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_mobilev2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_mobilev2.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_mobilev2.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

#%% for the custom wrn model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_wrn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_wrn.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_wrn.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
'''
now that all the models are trained on the second level pretraining data,
make predictions with the test data.
here we show it for a single model, repeat for other models
'''
#%%
vgg16_model = load_model('weights/vgg16_second_pretrained.03-0.8879.h5')
vgg16_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#%%
#measure performance on test data, first reset the test generator otherwise it gives wierd results
test_generator.reset()

#evaluate accuracy 
custom_y_pred = vgg16_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1) #if not absolutely divisible, otherwise remove 1

#save the predictions
np.savetxt('weights/vgg16_second_y_pred.csv',custom_y_pred,fmt='%f',delimiter = ",")
np.savetxt('weights/Y_test.csv',Y_test,fmt='%i',delimiter = ",")

#%%
#measure performance
accuracy = accuracy_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The test accuracy of the Custom model is: ', accuracy)

prec = precision_score(Y_test,custom_y_pred.argmax(axis=-1), 
                       average='weighted') 
print('The precision of the Custom model is: ', prec)

rec = recall_score(Y_test,custom_y_pred.argmax(axis=-1), 
                   average='weighted')
print('The recall of the Custom model is: ', rec)

f1 = f1_score(Y_test,custom_y_pred.argmax(axis=-1), 
              average='weighted')
print('The f1-score of the Custom model is: ', f1)

mat_coeff = matthews_corrcoef(Y_test,custom_y_pred.argmax(axis=-1))
print('The MCC of the Custom model is: ', mat_coeff)

#Cohen’s kappa: a statistic that measures inter-annotator agreement.
kappa = cohen_kappa_score(Y_test,
                          custom_y_pred.argmax(axis=-1))
print('The cohen kappa score of the Custom model is: ', kappa)

#%% print classification report and plot confusion matrix
import itertools

target_names = ['Bacterial', 'Non-COVID19_viral', 'Normal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix using scikit learn
plt.figure(figsize=(15,10), dpi=300)
plot_confusion_matrix(cnf_matrix, classes=target_names)
plt.show()

#%%
#plot ROC curves

fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test1[:, i], custom_y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test1.ravel(), custom_y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

#%%
# Zoom in view of the upper left corner.
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

#%%
#compute precision-recall curves
colors = cycle(['red', 'blue', 'green', 'cyan', 'teal'])

plt.figure(figsize=(15,10), dpi=300)
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test1[:, i],
                                                        custom_y_pred[:, i])
    average_precision[i] = average_precision_score(Y_test1[:, i], custom_y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test1.ravel(),
   custom_y_pred.ravel())
average_precision["micro"] = average_precision_score(Y_test1, custom_y_pred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(num_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.05)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class PR curve')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()

#%%
'''
Now that we are done with the second level of pretraining, and stored the best model
weights, we fine-tune these models 
toward classifying CXRs as showing COVID19 viral disease manifestions or normal lungs.
During this training step, the datasets are split 
into 80% for training and 20% for testing. We randomly allocated 
10% of the training data for validation. 
'''
#%% #data description
img_width, img_height = 256,256
train_data_dir = "../covid_finetuning/train"
test_data_dir = "../covid_finetuning/test"
epochs = 16
batch_size = 16
num_classes = 2 # COVID-19 + AND Normal
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
#define data generators
datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) 

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'training')

validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'validation')

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle = False)

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#true labels
Y_test=test_generator.classes
print(Y_test.shape)

#convert test labels to categorical
Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)

#%%
'''
initialize the models from the second level of pretraining. 
The models are truncated at their deepest convolutional layer 
and appended with (i) GAP layer, (ii) dropout layer (ratio = 0.5), and 
(iii) dense layer with Softmax activation to output class probabilities
for classifying CXRs as shown COVID-19 + or normal lungs   
'''
#%%
resnet18_cnn = load_model('weights/resnet18_second_pretrained.06-0.8936.h5')
resnet18_cnn.summary()

#%%
base_model_resnet18=Model(inputs=resnet18_cnn.input,
                        outputs=resnet18_cnn.get_layer('extra_conv_resnet18').output)
x = base_model_resnet18.output
x = GlobalAveragePooling2D()(x)              
x = Dropout(0.5)(x)                     
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_resnet18 = Model(inputs=base_model_resnet18.input, 
                    outputs=predictions, 
                    name = 'resnet18_new_covid_finetuned')
model_resnet18.summary()

#%%
xception_cnn = load_model('weights/xception_second_pretrained.03-0.9050.h5')
xception_cnn.summary()

#%%
base_model_xception=Model(inputs=xception_cnn.input,
                        outputs=xception_cnn.get_layer('extra_conv_xception').output)
x = base_model_xception.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_xception = Model(inputs=base_model_xception.input, 
                    outputs=predictions, 
                    name = 'xception_new_covid_finetuned')
model_xception.summary()

#%%
vgg16_cnn = load_model('weights/vgg16_second_pretrained.03-0.8879.h5')
vgg16_cnn.summary()

#%%
base_model_vgg16=Model(inputs=vgg16_cnn.input,
                        outputs=vgg16_cnn.get_layer('extra_conv_vgg16').output)
x = base_model_vgg16.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg16 = Model(inputs=base_model_vgg16.input, 
                    outputs=predictions, 
                    name = 'vgg16_new_covid_finetuned')
model_vgg16.summary()

#%%
vgg19_cnn = load_model('weights/vgg19_second_pretrained.02-0.8922.h5')
vgg19_cnn.summary()

#%%
base_model_vgg19=Model(inputs=vgg19_cnn.input,
                        outputs=vgg19_cnn.get_layer('extra_conv_vgg19').output)
x = base_model_vgg19.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg19 = Model(inputs=base_model_vgg19.input, 
                    outputs=predictions, 
                    name = 'vgg19_new_covid_finetuned')
model_vgg19.summary() 

#%%    
iv3_cnn = load_model('weights/iv3_second_pretrained.05-0.9277.h5')
iv3_cnn.summary()

#%%
base_model_iv3=Model(inputs=iv3_cnn.input,
                        outputs=iv3_cnn.get_layer('extra_conv_iv3').output)
x = base_model_iv3.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_iv3 = Model(inputs=base_model_iv3.input, 
                    outputs=predictions, 
                    name = 'iv3_new_covid_finetuned')
model_iv3.summary() 

#%%
densenet_cnn = load_model('weights/densenet_second_pretrained.06-0.9177.h5')
densenet_cnn.summary()

#%%
base_model_densenet=Model(inputs=densenet_cnn.input,
                        outputs=densenet_cnn.get_layer('extra_conv_densenet').output)
x = base_model_densenet.output    
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_densenet = Model(inputs=base_model_densenet.input, 
                    outputs=predictions, 
                    name = 'densenet_new_covid_finetuned')
model_densenet.summary() 

#%%
mobilev2_cnn = load_model('weights/mobilev2_second_pretrained.02-0.9121.h5')
mobilev2_cnn.summary()

#%%
base_model_mobilev2=Model(inputs=mobilev2_cnn.input,
                        outputs=mobilev2_cnn.get_layer('extra_conv_mobilev2').output)

x = base_model_mobilev2.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_mobilev2 = Model(inputs=base_model_mobilev2.input, 
                    outputs=predictions, 
                    name = 'mobilev2_new_covid_finetuned')
model_mobilev2.summary()

#%%
nasnet_cnn = load_model('weights/nasnet_second_pretrained.07-0.9163.h5')
nasnet_cnn.summary()

#%%
base_model_nasnet=Model(inputs=nasnet_cnn.input,
                        outputs=nasnet_cnn.get_layer('extra_conv_nasnet').output)
x = base_model_nasnet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
model_nasnet = Model(inputs=base_model_nasnet.input, 
                    outputs=predictions, 
                    name = 'nasnet_new_covid_finetuned')
model_nasnet.summary()

#%%

wrn_cnn = load_model('weights/wrn.05-0.7007.h5')
wrn_cnn.summary()

#%%
base_model_wrn=Model(inputs=wrn_cnn.input,
                        outputs=nasnet_cnn.get_layer('global_average_pooling2d').output)
x = base_model_wrn.output
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
model_wrn = Model(inputs=base_model_wrn.input, 
                    outputs=predictions, 
                    name = 'wrn_new_covid_finetuned')
model_wrn.summary()

#%% 
''' 
Train and validate these customized models on the COVID-19 data
'''
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_vgg16.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_vgg16.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 
#see if the steps per epochs and validation steps are absolutely divisble, otherwise add 1
 
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg19.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_vgg19.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_vgg19.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_iv3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_iv3.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_iv3.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_xception.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_xception.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_xception.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_densenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_densenet.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_densenet.fit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_nasnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_nasnet.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_nasnet.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 


#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_resnet18.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_resnet18.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_resnet18.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_mobilev2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_mobilev2.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_mobilev2.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

#%% for the custom wrn model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_wrn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_wrn.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_wrn.fit_generator(train_generator, 
                                     steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
'''
now that all the models are fine-tuned for the COVID-19 classification task,
we make predictions with the test data.
Here we show it for a single model, repeat for other models
'''
#%%
vgg16_model = load_model('weights/vgg16_new_covid_finetuned_8020.01-0.8681.h5')
vgg16_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#%%
#measure performance on test data, first reset the test generator otherwise it gives wierd results
test_generator.reset()

#evaluate accuracy 
custom_y_pred = vgg16_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1) #if not absolutely divisible, otherwise remove 1

#save the predictions
np.savetxt('weights/vgg16_covid_y_pred.csv',custom_y_pred,fmt='%f',delimiter = ",")
np.savetxt('weights/Y_test.csv',Y_test,fmt='%i',delimiter = ",")

#%%
#measure performance
accuracy = accuracy_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The test accuracy of the Custom model is: ', accuracy)

prec = precision_score(Y_test,custom_y_pred.argmax(axis=-1), 
                       average='micro') 
print('The precision of the Custom model is: ', prec)

rec = recall_score(Y_test,custom_y_pred.argmax(axis=-1), 
                   average='micro')
print('The recall of the Custom model is: ', rec)

f1 = f1_score(Y_test,custom_y_pred.argmax(axis=-1), 
              average='micro')
print('The f1-score of the Custom model is: ', f1)

mat_coeff = matthews_corrcoef(Y_test,custom_y_pred.argmax(axis=-1))
print('The MCC of the Custom model is: ', mat_coeff)

#Cohen’s kappa: a statistic that measures inter-annotator agreement.
kappa = cohen_kappa_score(Y_test,
                          custom_y_pred.argmax(axis=-1))
print('The cohen kappa score of the Custom model is: ', kappa)

#%% print classification report and plot confusion matrix
import itertools

target_names = ['COVID-19 +', 'Normal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix using scikit learn
plt.figure(figsize=(15,10), dpi=300)
plot_confusion_matrix(cnf_matrix, classes=target_names)
plt.show()

#%%
#plot ROC curves

fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test1[:, i], custom_y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test1.ravel(), custom_y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

#%%
# Zoom in view of the upper left corner.
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

#%%
#compute precision-recall curves

colors = cycle(['red', 'blue', 'green', 'cyan', 'teal'])

plt.figure(figsize=(15,10), dpi=300)
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test1[:, i],
                                                        custom_y_pred[:, i])
    average_precision[i] = average_precision_score(Y_test1[:, i], custom_y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test1.ravel(),
   custom_y_pred.ravel())
average_precision["micro"] = average_precision_score(Y_test1, custom_y_pred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(num_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.05)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class PR curve')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()

#%% 
'''
lets perform  t-SNE visualization for the best performing fine-tuned
model twoard COVID-19 detection task. here, we embed the 1024
dimensional feature space (features extracted from the deepest convolutional layer)
into 2- dimensions and visualize how the features for the normal
and COVID-19 classes are clustered in the 2-D feature space
'''
#%% load the best model finetuned on covid
base_model = load_model ('weights/covid_finetuned/resnet18_covid_finetuned_8020.02-0.8958.h5')
base_model.summary()

#%%
#create the new model; extract from the GAP layer
model = Model(inputs=base_model.input, 
                    outputs=base_model.get_layer('global_average_pooling2d_19').output)
model.summary()
#compile the model
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

#%%
'''
We will define the image features. The definition runs the 
given image_files to model and returns the weights 
(filters) as a 1, 1024 dimension vector from the output of the GAP layer. 
Since the model was trained as a image of 256x256, every new image is 
required to go through the same transformation.
'''
def get_image_features(image_file_name):
    
    image_features = np.zeros((1, 1024)) # comes from GAP
    im = cv2.resize(cv2.imread(image_file_name), (256,256))
    im = im.astype(np.float32, copy=False) # shape of im = (256,256,3)
    im = np.expand_dims(im, axis=0)  # shape of im = (1,256,256,3)
    image_features[0,:] = model.predict(im)[0]
    return image_features

#%%
'''
Now lets get the path to the data and compute the image features. 
In our case, the main folder will be 'test' and the 
classes will be 'COVID-19 +' and 'normal'. 
The features will be extracted from the test data and shown as embeddings. 
'''

PATH=os.getcwd()
data_path = PATH + './test'
data_dir_list = os.listdir(data_path)
image_features_list=[] #create a list to store image features

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Extracting Features of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        image_features=get_image_features(data_path + '/'+ dataset + '/'+ img )
        image_features_list.append(image_features)
    
    
image_features_arr=np.asarray(image_features_list)
image_features_arr = np.rollaxis(image_features_arr,1,0)
image_features_arr = image_features_arr[0,:,:]

np.savetxt('feature_vectors.txt',image_features_arr)
pickle.dump(image_features_arr, open('feature_vectors.pkl', 'wb'))

#%%
'''
Now that we have extracted the features, 
we will now proceed to visualize the feature embeddings using Tensorboard. 
The embeddings shows not only the features 
but also the images corresponding to those respective features that are overlayed on
the features. this collects the details of the data
'''
# path where the embedding logs will be saved
PATH = os.getcwd()
LOG_DIR = PATH+ '/embedding'
data_path = PATH + '/test'
data_dir_list = os.listdir(data_path)
img_data=[]
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(256,256)) 
        #this size can be anything depends on how clearly we want to 
        #visualize the images on the tensorbord. 
        img_data.append(input_img_resize)

#convert the images to array  
img_data = np.array(img_data)

#%%
#Let us now load the extracted features and store them in a tensorflow variable

feature_vectors = np.loadtxt('feature_vectors.txt')
print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])

num_of_samples=feature_vectors.shape[0]
num_of_samples_each_class = 72 #COVID-19 and Normal test images count

features = tf.Variable(feature_vectors, name='features')

#%%
#let us give lables for the classes and the names for the categories. 
#See the order in which the images where loaded before.

y = np.ones((num_of_samples,),dtype='int64')
y[0:72]=0 #covid abnormal
y[72:]=1 #normal cxrs
names = ['COVID-19','Normal']

#%%
'''
Let us store the class labels and the numbers now. 
This gives information on which features belong to which
labels and it keeps both the names and the numbers.
They assume equal number of images across each classes here. 
'''
metadata_file = open(os.path.join(LOG_DIR, 'metadata_2_classes.tsv'), 'w')
metadata_file.write('Class\tName\n')
k=72 # num of samples in each class
j=0
for i in range(num_of_samples):
    c = names[y[i]]
    if i%k==0:
        j=j+1
    metadata_file.write('{}\t{}\n'.format(j,c))
metadata_file.close()

#%%
'''
The following definition file creates the combined image 
along with any necessary padding. The input arguments are: 
data: NxHxWX3 tensor containing the images and it returns data: 
Properly shaped HxWx3 image with any necessary padding.All the 
images of all the classes will be added to the combined image.
'''
def images_to_combine(data):
 
    if len(data.shape) == 3: 
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

combine = images_to_combine(img_data)
cv2.imwrite(os.path.join(LOG_DIR, 'combined_2_classes.png'), combine)

#%%
#Now lets create the logs for the features along with the labels to visualize the embeddings.
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images_2_classes.ckpt'))
    
    config = projector.ProjectorConfig()
    
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_2_classes.tsv')
    
    # Comment out if you don't want combined images
    embedding.combine.image_path = os.path.join(LOG_DIR, 'combined_2_classes.png')
    embedding.combine.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

#%%
'''
To run the embeddings launch tensor board
tensorboard --logdir=embedding --port=6006
## Please make sure there is no gap between the name of your directory-
for e.g.- folder name will not work it has to be folder_name
Then open localhost:6006 in a browser
Then go to the embedding options in Tensorboard
'''

#%%
'''
we need to compare the performance of the fine-tuned models
to the baseline models. i.e. the imagenet pretrained models with the ImagNet weights
that are directly retrained on the COVID-19 data.
Here, we show a exmaple with one model, repeat for other ImageNet pretrained models
'''
#%% forked from https://github.com/qubvel/classification_models
ResNet18, preprocess_input = Classifiers.get('resnet18')
# build model
resnet18_cnn = ResNet18(input_tensor=model_input, 
                        weights='imagenet', include_top=False)
resnet18_cnn.summary()
x = resnet18_cnn.output
x = GlobalAveragePooling2D()(x)              
x = Dropout(0.5)(x)                     
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_resnet18_baseline = Model(inputs=resnet18_cnn.input, 
                    outputs=predictions, 
                    name = 'resnet18_baseline_covid')
model_resnet18_baseline.summary()

#%%
#train the model on teh COVID-19 data and record perfomance as before
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_resnet18_baseline.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
filepath = 'weights/' + model_resnet18_baseline.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_weights_only=False, save_best_only=True, 
                             mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_acc', 
                               patience=4, verbose=1, mode='max')
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]


#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
history = model_resnet18_baselinefit_generator(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1) 

#%%
'''
use the codes above for the fine-tuned models to evalaute the performance
with the test data and record performance metrics, AUC, and PR curves.
We observed that the baseline performance is worse compared to the 
fine-tuned models
'''

#%%
'''
Now we use our in-house visualization CRM tool to loclaize the salient ROI
predicted by the individual fine-tuned and baselne models to see if they have
learned the imlicit rules well. Here, we show an exmaple with one fine-tuned model,
the process shall be repated with other model weights.
'''
#%%
#load model
vgg16_custom_model = load_model('weights/vgg16_covid_finetuned_8020.01-0.8681.h5')
vgg16_custom_model.summary()
#compile the model
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True) #optimize to your requirements
vgg16_custom_model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%
#path for the image to visualize
from keras.preprocessing import image
img_path = 'patient3_3.png' #an image with COVID-19 disease patterns
img = image.load_img(img_path)

#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255 

#predict on the image
preds = vgg16_custom_model.predict(x)[0]
print(preds)

#%%
#this is the custom definition for our CRM visualization

def Generate_CRM_2Class(thisModel, thisImg_path, Threshold):  # generate Class Revlevance Map (CRM)
    try:
        # preprocess input image      
        width, height = thisModel.input.shape[1:3].as_list()
        original_img = cv2.imread(thisImg_path) 
        resized_original_image = cv2.resize(original_img, (width,height))        
    
        input_image = img_to_array(resized_original_image)
        input_image = np.array(input_image, dtype="float") /255.0       
        input_image = input_image.reshape((1,) + input_image.shape)
    
        class_weights = thisModel.layers[-1].get_weights()[0]
    
        get_output = K.function([thisModel.layers[0].input], [thisModel.layers[-4].output, 
                                 thisModel.layers[-1].output])
        [conv_outputs, predictions] = get_output([input_image])
        conv_outputs = conv_outputs[ 0, :, :, :]     
        final_output = predictions   
        
        wf0 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])    
        wf1 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])    
    
        for i, w in enumerate(class_weights[:, 0]):     
            wf0 += w * conv_outputs[:, :, i]           
        S0 = np.sum(wf0)           # score at node 0 in the final output layer
        for i, w in enumerate(class_weights[:, 1]):     
            wf1 += w * conv_outputs[:, :, i]             
        S1 = np.sum(wf1)           # score at node 1 in the final output layer
    
        #Calculate incremental MSE
        iMSE0 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        iMSE1 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    
        row, col = wf0.shape
        for x in range (row):
                for y in range (col):
                        tmp0 = np.array(wf0)
                        tmp0[x,y] = 0.                   # remove activation at the spatial location (x,y)
                        iMSE0[x,y] = (S0 - np.sum(tmp0)) ** 2
    
                        tmp1 = np.array(wf1)
                        tmp1[x,y] = 0.                  
                        iMSE1[x,y] = (S1 - np.sum(tmp1)) ** 2
         
      
        Final_crm = iMSE0 + iMSE1       # consider both positive and negative contribution
    
        Final_crm /= np.max(Final_crm)    # normalize
        resized_Final_crm = cv2.resize(Final_crm, (height, width)) # upscaling to original image size

        The_CRM = np.array(resized_Final_crm)
        The_CRM[np.where(resized_Final_crm < Threshold)] = 0.  # clean-up (remove data below threshold)

        return[resized_original_image, final_output, resized_Final_crm, The_CRM]
    except Exception as e:
        raise Exception('Error from Generate_CRM_2Class(): ' + str(e))

#%%
# a code snippet taken from stackoverflow to store images with high dpi

def writePNGwithdpi(im, filename, dpi=(72,72)):
   """Save the image as PNG with embedded dpi"""

   # Encode as PNG into memory
   retval, buffer = cv2.imencode(".png", im)
   s = buffer.tostring()

   # Find start of IDAT chunk
   IDAToffset = s.find(b'IDAT') - 4
   pHYs = b'pHYs' + struct.pack('!IIc',int(dpi[0]/0.0254),int(dpi[1]/0.0254),b"\x01" ) 
   pHYs = struct.pack('!I',9) + pHYs + struct.pack('!I',zlib.crc32(pHYs))
   with open(filename, "wb") as out:
      out.write(buffer[0:IDAToffset])
      out.write(pHYs)
      out.write(buffer[IDAToffset:])

#%%
InImage1, OutScores1, aCRM_Img1, tCRM_Img1 = Generate_CRM_2Class(vgg16_custom_model,
                                                                 img_path, 0.1) 
plt.figure() # create a figure with the default size 
plt.imshow(InImage1)
aHeatmap = cv2.applyColorMap(np.uint8(255*aCRM_Img1), cv2.COLORMAP_JET)
aHeatmap[np.where(aCRM_Img1 < 0.2)] = 0   
superimposed_img = aHeatmap * 0.4 + InImage1 #0.4 here is a heatmap intensity factor.
#if we have to increse the DPI and write to disk
writePNGwithdpi(superimposed_img, "vgg16_covid_roi.png", (300,300))

#%%
'''
Now we have all the custom and pretrained models fine-tuned for the
COVID-19 detection task. we now proceed to perform ensembles of these models
to evalute for an improvement in performance.
The following ensemble methods are attempted:
majority voting
simple averaging
weighted averaging
After all these experiments, we found the best 7 fine-tuned models to perform the 
ensemble that demonstrated superior performance in COVID-19 detection. 
the top-7 are Xception, Inception-V3, DenseNet-121, VGG-19,
VGG-16, NasNet-Mobile, and ResNet-18 (not in order)
Now we load each model and record its predictions
'''
#%%
#top-1: ResNet-18
resnet18_model = load_model('weights/resnet18_covid_finetuned_8020.02-0.8958.h5')
resnet18_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
resnet18_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
resnet18_y_pred = resnet18_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#top-2: MobileNet-V2
mobilev2_model = load_model('weights/mobilev2_covid_finetuned_8020.03-0.8750.h5')
mobilev2_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
mobilev2_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
mobilev2_y_pred = mobilev2_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#top-3: Densenet-121
densenet_model = load_model('weights/densenet_covid_finetuned_8020.08-0.8750.h5')
densenet_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
densenet_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
densenet_y_pred = densenet_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#top-4: xception
xception_model = load_model('weights/xception_covid_finetuned_8020.13-0.8681.h5')
xception_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
xception_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
xception_y_pred = xception_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#top-5: VGG-16
vgg16_model = load_model('weights/vgg16_covid_finetuned_8020.01-0.8681.h5')
vgg16_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
vgg16_y_pred = vgg16_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#top-6: VGG-19
vgg19_model = load_model('weights/vgg19_covid_finetuned_8020.01-0.8611.h5')
vgg19_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
vgg19_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
vgg19_y_pred = vgg19_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#top-7: Inception-V3
inceptionv3_model = load_model('weights/iv3_covid_finetuned_8020.02-0.8611.h5')
inceptionv3_model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
inceptionv3_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

#measure performance on test data, 
test_generator.reset()
inceptionv3_y_pred = inceptionv3_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)

#%%
#perform majority voting
#lets do a dummy assignment of the predictions
xception_y_pred1 = xception_y_pred
inceptionv3_y_pred1 = inceptionv3_y_pred
densenet_y_pred1 = densenet_y_pred
vgg19_y_pred1 = vgg19_y_pred
vgg16_y_pred1 = vgg16_y_pred
mobilev2_y_pred1 = mobilev2_y_pred
resnet18_y_pred1 = resnet18_y_pred

#print the shape of the predictions
print("The shape of xception predcition  = ", 
      xception_y_pred1.shape)
print("The shape of inception predcition  = ", 
      inceptionv3_y_pred1.shape)
print("The shape of densenet predcition  = ", 
      densenet_y_pred1.shape)
print("The shape of vgg19 predcition  = ", 
      vgg19_y_pred1.shape)
print("The shape of vgg16 predcition  = ", 
      vgg16_y_pred1.shape)
print("The shape of MobileNet-V2 predcition  = ", 
      mobilev2_y_pred1.shape)
print("The shape of resnete18 predcition  = ", 
      resnet18_y_pred1.shape)

#%%
#compute argmax
xception_y_pred1 = xception_y_pred1.argmax(axis=-1)
inceptionv3_y_pred1 = inceptionv3_y_pred1.argmax(axis=-1)
densenet_y_pred1 = densenet_y_pred1.argmax(axis=-1)
vgg19_y_pred1 = vgg19_y_pred1.argmax(axis=-1)
vgg16_y_pred1 = vgg16_y_pred1.argmax(axis=-1)
mobilev2_y_pred1 = mobilev2_y_pred1.argmax(axis=-1)
resnet18_y_pred1 = resnet18_y_pred1.argmax(axis=-1)

#%%
#max voting begins
#top-3 ensemble:
max_voting_3_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_3_pred = np.append(max_voting_3_pred, 
                                statistics.mode([resnet18_y_pred1[i],
                                                 mobilev2_y_pred1[i],
                                                 densenet_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_3_pred1=to_categorical(max_voting_3_pred, num_classes=num_classes, dtype='float32')
print(max_voting_3_pred1.shape)

#accuracy    
ensemble_model_3_max_voting_accuracy = accuracy_score(Y_test,max_voting_3_pred)
print("The max voting accuracy of the ensemble model is  = ", 
      ensemble_model_3_max_voting_accuracy)

#save the predictions
np.savetxt('weights/max_voting_3_y_pred.csv',max_voting_3_pred1,fmt='%f',delimiter = ",")

#%%
# top-5 max voting ensemble:
max_voting_5_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_5_pred = np.append(max_voting_5_pred, 
                                statistics.mode([xception_y_pred1[i],
                                                 resnet18_y_pred1[i],
                                                 densenet_y_pred1[i],
                                                 mobilev2_y_pred1[i],
                                                 vgg16_y_pred1[i]                                                 
                                                ]))

max_voting_5_pred1=to_categorical(max_voting_5_pred, num_classes=num_classes, dtype='float32')
print(max_voting_5_pred1.shape)
ensemble_model_5_max_voting_accuracy = accuracy_score(Y_test,max_voting_5_pred)
print("The max voting accuracy of the ensemble model is  = ", 
      ensemble_model_5_max_voting_accuracy)

#save the predictions
np.savetxt('weights/max_voting_5_y_pred.csv',max_voting_5_pred1,fmt='%i',delimiter = ",")

#%%
#op-7 max voting ensemble:
max_voting_7_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_7_pred = np.append(max_voting_7_pred, 
                                statistics.mode([xception_y_pred1[i],
                                                 inceptionv3_y_pred1[i],
                                                 densenet_y_pred1[i],
                                                 vgg19_y_pred1[i],
                                                 vgg16_y_pred1[i],
                                                 mobilev2_y_pred1[i],
                                                 resnet18_y_pred1[i]                                                 
                                                ]))
    
max_voting_7_pred1=to_categorical(max_voting_7_pred, 
                                  num_classes=num_classes, dtype='float32')
print(max_voting_7_pred1.shape)
ensemble_model_7_max_voting_accuracy = accuracy_score(Y_test,max_voting_7_pred)
print("The max voting accuracy of the ensemble model is  = ", 
      ensemble_model_7_max_voting_accuracy)

#save the predictions
np.savetxt('weights/max_voting_7_y_pred.csv',max_voting_7_pred1,fmt='%i',delimiter = ",")

#%%
#simple averaging
#top-3
average_pred_3=(resnet18_y_pred + densenet_y_pred + mobilev2_y_pred)/3
#top-5
average_pred_5=(resnet18_y_pred + densenet_y_pred + mobilev2_y_pred + xception_y_pred + \
                vgg16_y_pred)/5
#top-7
average_pred_7=(resnet18_y_pred + densenet_y_pred + mobilev2_y_pred + xception_y_pred + \
                vgg16_y_pred + vgg19_y_pred + inceptionv3_y_pred)/7

#compute ccuracy
#top-3:
ensemble_model_3_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_3.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_3_averaging_accuracy)

#top-5:
ensemble_model_5_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_5.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_5_averaging_accuracy)

#top-7
ensemble_model_7_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_7.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_7_averaging_accuracy)

#save the predictions
np.savetxt('weights/averaging_3_y_pred.csv',average_pred_3,fmt='%i',delimiter = ",")
#save the predictions
np.savetxt('weights/averaging_5_y_pred.csv',average_pred_5,fmt='%i',delimiter = ",")
#save the predictions
np.savetxt('weights/averaging_7_y_pred.csv',average_pred_7,fmt='%i',delimiter = ",")

#%%
#weighted averaging:
'''
Here, we calcualte the optimal weights for the models 
predictions through a constrained minimization process of the logarithmic loss 
function
'''
#%%
preds = []
test_generator.reset()
resnet18_y_pred = resnet18_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)
preds.append(resnet18_y_pred)
test_generator.reset()
mobilev2_y_pred = mobilev2_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)
preds.append(mobilev2_y_pred)
test_generator.reset()
densenet_y_pred = densenet_model.predict_generator(test_generator,
                                        nb_test_samples // batch_size + 1, verbose=1)
preds.append(densenet_y_pred)    

# keep appending the pedictions for top-5 and top-7 ensembles
#%%
#define a custom function to measure weighted accuracy

def calculate_weighted_accuracy(prediction_weights):
    weighted_predictions = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = Y_test1.argmax(axis=-1)
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

#lets assume equal weights for the model predictions to begin with
prediction_weights = [1. / 3] * 3 #modify for top-5 and top-7
calculate_weighted_accuracy(prediction_weights)

#%%
# Create the loss metric 
def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((nb_test_samples, num_classes), 
                                dtype='float32')
    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction
    return log_loss(Y_test1, final_prediction)

best_acc = 0.0
best_weights = None

# Parameters for optimization
constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(preds)

#%%
'''
now we determine how much weights we have to give
for each ensemble model based on the log loss functions
the process is repeated for 50 times to find the best combination
of weights for the ensemble models that results in
the highest accuracy and lowest loss
'''

NUM_TESTS = 50 
# Check for NUM_TESTS times
for iteration in range(NUM_TESTS):
    # Random initialization of weights for the top-3 model predictions
    prediction_weights = np.random.random(3) #change for top-5 and top-7
    
    # Minimise the loss 
    result = minimize(log_loss_func, prediction_weights, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    print('Best Ensemble Weights: {weights}'.format(weights=result['x']))
    
    weights = result['x']
    weighted_predictions = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')    
    # Calculate weighted predictions
    for weight, prediction in zip(weights, preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = Y_test1.argmax(axis=-1)
    # Calculate weight prediction accuracy
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
    print("Iteration %d: Error : " % (iteration + 1), error)
    
    # Save current best weights 
    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights
        
    print()

print("Best Accuracy : ", best_acc)
print("Best Weights : ", best_weights)
calculate_weighted_accuracy(best_weights)

#%%
'''this gives the values for the best accuracy and best weights
for the model predictions. Use them to calcualte accuracy and error.
'''

prediction_weights = np.array([0.6357, 0.1428, 0.2216]) #varies depending on your study, modify for top-5 and top-7
weighted_predictions_3 = np.zeros((nb_test_samples, num_classes), 
                                dtype='float32')
for weight, prediction in zip(prediction_weights, preds):
    weighted_predictions_3 += weight * prediction
yPred_bestweighted_3 = np.argmax(weighted_predictions_3, axis=1)
yTrue = Y_test1.argmax(axis=-1)
accuracy = metrics.accuracy_score(yTrue, yPred_bestweighted_3) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

np.savetxt('weighted_averaging_3_y_pred.csv',weighted_predictions_3,fmt='%f',delimiter = ",")

#%%
'''repeat the above process for the top-5 and top-7
models to record the weighted predictions
'''
#%%
'''
for each of the above predictions (majority, siple averaing, and weighted averaging), 
measure the performance, plot confusion matrix, plot roc curves, and PR curves
using the codes given below. we have shown an instance of using the predictions from
simple averaging using the top-3 models. modify the prediction
variable and record performance for other ensemble methods and options.
'''
#%%
accuracy = accuracy_score(Y_test1.argmax(axis=-1),
                          average_pred_3.argmax(axis=-1))
prec = precision_score(Y_test,
                       average_pred_3.argmax(axis=-1), 
                       average='micro') 
rec = recall_score(Y_test,
                   average_pred_3.argmax(axis=-1), 
                   average='micro')
f1 = f1_score(Y_test,
              average_pred_3.argmax(axis=-1), 
              average='micro')
mat_coeff = matthews_corrcoef(Y_test,
                              average_pred_3.argmax(axis=-1))
kappa = cohen_kappa_score(Y_test,
                          average_pred_3.argmax(axis=-1))

#%% print classification report and plot confusion matrix
target_names = ['COVID-19 +', 'Normal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            average_pred_3.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              average_pred_3.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix using scikit learn
plt.figure(figsize=(15,10), dpi=300)
plot_confusion_matrix(cnf_matrix, classes=target_names)
plt.show()

#%%
#plot ROC curves

fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test1[:, i], average_pred_3[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test1.ravel(), 
                                          average_pred_3.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Top-N ensembles')
plt.legend(loc="lower right")
plt.show()

#%%
# Zoom in view of the upper left corner.
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

#plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC FOR TOP-3 SIMPLE AVERAGING ENSEMBLE')
plt.legend(loc="lower right")
plt.show()

#%%
#compute precision-recall curves
colors = cycle(['red', 'blue', 'green', 'cyan', 'teal'])

plt.figure(figsize=(15,10), dpi=300)
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test1[:, i],
                                                        average_pred_3[:, i])
    average_precision[i] = average_precision_score(Y_test1[:, i], 
                                                   average_pred_3[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test1.ravel(),
                                                                average_pred_3.ravel())
average_precision["micro"] = average_precision_score(Y_test1, 
                                                     average_pred_3,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(num_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.05)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class PR curve')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()

#%%
'''
with this, the model training and evaluation for individual and 
ensemble models is completed. we furhter use the trained models 
toward evaluating the localization performance
'''
