'''
This part of the code describes the visualization studies
performed toward localizing COVID-19 viral disease 
specific manifestations in CXRs.
The fine-tuned CNN models toward the COVID-19 detection task are
evalauted for their performance in localizing the disease ROI.
We used out in-house visualization tool for class-selective relevance mapping (CRM)
https://www.mdpi.com/2075-4418/9/2/38/htm to visualize disease ROI
using individual models and their ensembles.
Using this code base, we do the following:
(i) perform localization studies using ensemble CRM; 
(ii) compute PR curves for the model versus radioloigts 
and model versus staple generated consensus annotation; 
(iii) analyse inter-reader variability using kappa and other measures.  
'''
#%%
#import libraries
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, SeparableConv2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D,Flatten,Average, Dropout
import time
from skimage.segmentation import mark_boundaries
import statistics
from keras import applications
from scipy import interp
import cv2
import imutils
from copy import deepcopy
import json
import glob
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from keras.preprocessing import image
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
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import ndimage
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math
import os
import glob
import h5py
import csv
import json
from collections import namedtuple
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
import SimpleITK as sitk
import numpy as np
from skimage.measure import label, regionprops
from skimage import io
from scipy.ndimage.morphology import binary_fill_holes
import os
import glob
import pandas as pd
import json
import cv2
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from scipy.ndimage.interpolation import zoom
import numpy as np
from keras.backend import tensorflow_backend
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

#%%
#define a custom model to store images at high resolution
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
'''
The JSON markings received from the experts (who annoatated the images using
VGG annotator tool) are converted to CSV files for further use
'''
#%%
with open('expert1_markings.json') as f:
    data = json.load(f)
    
csv = open('expert1_annotations.csv','w') #modify for different experts annotations
csv.write("filename,size,x,y,width,height\n")

im_data = data["_via_img_metadata"]
for info in im_data:
    if info.startswith('patient'):
        filename = im_data[info]['filename']
        size = im_data[info]['size']
        for region_info in im_data[info]['regions']:
            x = region_info['shape_attributes']['x']
            y = region_info['shape_attributes']['y']
            width = region_info['shape_attributes']['width']
            height = region_info['shape_attributes']['height']
            csv.write(filename+','+str(size)+','+str(x)+','+str(y)+','+str(width)+','+str(height)+'\n')
    
csv.close()
# The JSON files received from the two experts are converted to CSV files

#%%
'''
we will visualize the disease ROI using individual 
fine-tuned models toward COVID-19 detection.
Here, we show an instance of using the VGG-16 fine-tuned model
to localize COVID-19 disease-specific ROI in a sample CXR showing
COVID-19 disease manifestations.
'''
#%%
#define model
vgg16_custom_model = load_model('weights/vgg16_covid_finetuned_8020.01-0.8681.h5')
vgg16_custom_model.summary()
#compile the model
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True) #optimize to your requirements
vgg16_custom_model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%
#path to image to visualize
width,height = 256, 256
img_path = 'patient3_3.png'
img = image.load_img(img_path)
#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255 
#predict on the image
preds = vgg16_custom_model.predict(x)[0]
print(preds)

# This gives class level prediction probabilities

#%%
''' now, we define the function for our in-house
CRM visualization. The details of the algorithm can be found at 
https://www.mdpi.com/2075-4418/9/2/38/htm 
'''
#%%
def Generate_CRM_2Class(thisModel, thisImg_path, Threshold):             # generate Class Revlevance Map (CRM)
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
'''
The visualization function returns the resized_original_image, 
final_output, resized_Final_CRM, and thresholded CRM.
we threshold activations below 0.2 to remove noise
'''
InImage, OutScores, aCRM_Img, tCRM_Img = Generate_CRM_2Class(vgg16_custom_model,
                                                                 img_path, 0.2) 
plt.figure() 
plt.imshow(InImage)
aHeatmap = cv2.applyColorMap(np.uint8(255*aCRM_Img1), cv2.COLORMAP_JET)
aHeatmap[np.where(aCRM_Img1 < 0.2)] = 0   
superimposed_img = aHeatmap * 0.4 + InImage1 #0.4 here is a heatmap intensity factor.
writePNGwithdpi(superimposed_img, "vgg16_covid_roi.png", (300,300))

#%%
'''
Similarly, we generate class-relevance mapping for 
individual instances of CXRs using various fine-tuned models
toward COVID-19 detection
'''
#%%
'''
Next, we draw bounding boxes for the disease specific ROI for a sample CXR
by loading the CSV file of the experts to find the disease ROI annotations.
We load the CSV files of the two experts and also
the one generated by STAPLE algorithm that gives a consensus annotation
by combining the annotations of the two experts. 
'''

#load CSV
df = pd.read_csv('../annotation_expert1.csv') #modify for expert-2 and STAPLE generated consensus ROI
#load image
image = cv2.imread('patient3_3.png')
#give image name
im_name = 'patient3_3.png'

#loop through and draw bounding boxes
for i in range(len(df)):
    name = df.loc[i]['patientId']
    if name == im_name:
        # Start coordinate, represents the top left corner of rectangle
        start_point = (df.loc[i]['x_dis'],df.loc[i]['y_dis'])              
        # Ending coordinate, represents the bottom right corner of rectangle
        end_point = (df.loc[i]['x_dis']+df.loc[i]['width_dis']
                         ,df.loc[i]['y_dis']+df.loc[i]['height_dis'])   
        color = (255, 0, 0)
        thickness = 2  
        # Using cv2.rectangle() method
        image = cv2.rectangle(image, start_point, end_point, color, thickness) 
        #Displaying the image  
cv2.imwrite('image_with_roi.png', image) 

#%%
'''
STAPLE stands for Simultaneous Truth and Performance Level Estimation.
This algorithm estimates a reference standard from a set of annotations. 
It has been widely applied for the validation 
of annotation process, and to compare the performance of 
different algorithms and experts. It has also found application in the 
identification of a consensus annotation, by combination of the output 
of a group of experts annotations.
We installed SimpleITK for this process, it contains predefined
functions to run STAPLE algorithm.
'''
#%%
#custom function to get disease ROI annotaions

def get_mask(img_name, df, h, w):
    im_csv_np = df.loc[:,"patientId"].values
    idx = np.where(im_csv_np == img_name)
    if idx[0].shape[0]: # if there is a match shape[0] should 1, if not 0
        mask = np.zeros((len(idx[0]),h,w))
        for k,j in enumerate(idx[0]):
            i = j.item()
            mask[k,int(df.loc[i]['y_dis']):int(df.loc[i]['y_dis'])+int(df.loc[i]['height_dis']),
                        int(df.loc[i]['x_dis']):int(df.loc[i]['x_dis'])+int(df.loc[i]['width_dis'])] = 1.0
    else:
        mask = np.zeros((1,h,w))
    return mask

#%%

#get multi-dimensional mask
def multi_dim_mask(mask):
    labels = label(mask)
    maskf = np.zeros((np.max(labels),mask.shape[0],mask.shape[1]))
    for label_idx in range(np.max(labels)):
        maskf[label_idx][labels == label_idx+1] = 1.0
    return maskf

#%%
#STAPLE algorithm to generate reference masks
   
def ref_generator(masks):
    segmentations = []
    for m in masks:
        mask = m.astype(np.int8)
        sitk_m = sitk.GetImageFromArray(np.sum(mask,0))
        segmentations.append(sitk_m)
    # combines a single label from multiple segmentations, 
    #the label is user specified. The result of the
    # filter is the pixel's probability of belonging to the foreground. 
    #We then have to threshold the result to obtain
    # a reference binary segmentation.
    foregroundValue = 1
    threshold = 0.95
    reference_segmentation_STAPLE_probabilities = sitk.STAPLE(segmentations, foregroundValue) 
    # We use the overloaded operator to perform thresholding, 
    #another option is to use the BinaryThreshold function.
    reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > threshold
    ref_mask = sitk.GetArrayFromImage(reference_segmentation_STAPLE)
    ref_mask = binary_fill_holes(ref_mask).astype(np.float64)
    ref_mask_ = ref_mask.astype(np.uint8)
    ref_mask_[ref_mask_==1] = 255
    ref_mask = multi_dim_mask(ref_mask)
    return ref_mask, ref_mask_

#%%
#IOU functions to validate performance
def jaccard_index(a,b):
    intersection = np.sum(np.multiply(a,b))
    union = np.sum(a+b) - intersection
    return intersection/union

def dice_score(a,b):
    num = 2 * np.sum(np.multiply(a,b))
    den = np.sum(a) + np.sum(b)
    return num/den

#%%
#Kappa statistic measurement for later evaluation

def kappa_eval(ref,mask,thr, metric = 'jaccard'):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    if ref.any() or mask.any():
        for i in range(len(mask)):
            for j in range(len(ref)):
                if metric == 'jaccard':
                    iou_val = jaccard_index(mask[i],ref[j])
                else:
                    iou_val = dice_score(mask[i],ref[j])
                if iou_val >= thr:
                    TP += 1
                elif iou_val < thr and iou_val > 0:
                    FP += 1
        for k in range(len(mask)):
            if np.sum(np.multiply(mask[k],np.sum(ref,0))) == 0:
                FP += 1
        if len(ref) > TP+FP:
            FN += len(ref) - (TP+FP)
    else:
        TN = 1
    
    print("TP: {}, FP: {}, FN: {}, TN: {}".format(TP,FP,FN,TN))
    print("*"*20)
    return TP,FP,FN,TN

#%%

#define Kappa evaluation metric
def eval_metric(TP, FP, FN, TN):
    po = (TP+TN)/(TP+FP+FN+TN)
    p_true = ((TP+FN)*(TP+FP))/((TP+FP+FN+TN)**2)
    p_false = ((FP+TN)*(FN+TN))/((TP+FP+FN+TN)**2)
    pe = p_true + p_false
    if pe == 1:
        k = 1
    else:
        k = (po - pe)/(1-pe)
    return k

#%%
#function to overlay masks on the original image

def overlay(org_img,ref,mask):
    org_img = cv2.cvtColor(org_img,cv2.COLOR_GRAY2RGB)
    cm_overlay = org_img.copy()
    # Ref red contours
    for i in range(len(ref)):
        contours, _ = cv2.findContours(ref[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            cv2.drawContours(cm_overlay, contour, -1, (255, 0, 0), 2)    
    # test blue contours
    for j in range(len(mask)):
        contours2, _ = cv2.findContours(mask[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour2 in contours2:
            cv2.drawContours(cm_overlay, contour2, -1, (0, 0, 255), 1)   
    return cm_overlay

#%%
#define performance metrics
def sensitivity_metric(TP, FP, FN, TN):
    
    pnum = TP/(TP + FN)
    return pnum

def specificity_metric(TP, FP, FN, TN):
    
    num = TN/(TN + FP)
    return pnum

def ppv_metric(TP, FP, FN, TN):
    
    pnum = TP/(TP + FP)
    return pnum

def npv_metric(TP, FP, FN, TN):
    
    pnum = TN/(TN + FN)
    return pnum

#%%

if __name__ == '__main__':
    rad = 'program' # ['rad-1', 'rad-2', 'program']
    metric = 'jaccard' # ['jaccard','dice']
    iou_thr = 0.1 # [10 different IoU thresholds in the range (0.1 - 0.7)]
    
    filenames = glob.glob("../test/*.png")
    filenames.sort()
    # expert_1
    df1 = pd.read_csv('../expert_1.csv')
    # expert_2
    df2 = pd.read_csv('../expert_2.csv')
    # program
    #predictions generated by the top-3 ensembles using STAPLE-generated consensus ROI as standard
    df3 = pd.read_csv('../program.csv') 
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    csv = open('../bounding_box_test_STAPLE.csv','w')
    csv.write("patientId,x_dis,y_dis,width_dis,height_dis,Labels\n")
    
    for f in filenames:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        h,w = img.shape
        img_name = f.split(os.sep)[-1]
        print(img_name)
        mask1 = get_mask(img_name, df1, h, w)
        mask2 = get_mask(img_name, df2, h, w)
        mask3 = get_mask(img_name, df3, h, w) 
    
        #generate consensus using rad-1 and rad-2 annotations
        ref_mask,ref_mask_ = ref_generator(masks = (mask1,mask2))        

        ref_mask_overlay = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        for i in range(len(ref_mask)):
            props = regionprops(ref_mask[i].astype(np.int8))[0]
            df_csv = {"patientId": img_name,
                      "x_dis": props.bbox[1],
                      "y_dis": props.bbox[0],
                      "width_dis": abs(props.bbox[3]-props.bbox[1]),
                      "height_dis": abs(props.bbox[2]-props.bbox[0]),
                      "Labels": "COVID-19"}
            csv.write(img_name+','+str(props.bbox[1])+','+str(props.bbox[0])+','+str(abs(props.bbox[3]-props.bbox[1]))+','+str(abs(props.bbox[2]-props.bbox[0]))+','+'COVID-19'+'\n')
            contours, _ = cv2.findContours(ref_mask[i].astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                cv2.drawContours(ref_mask_overlay, contour, -1, (255, 0, 0), 3)
        
        if rad == 'rad-1':
            TP_,FP_,FN_, TN_ = kappa_eval(ref_mask, mask1, iou_thr, metric)
            cm_overlay = overlay(img, ref_mask, mask1)
        elif rad == 'rad-2':
            TP_,FP_,FN_, TN_ = kappa_eval(ref_mask, mask2, iou_thr, metric)
            cm_overlay = overlay(img, ref_mask, mask2)
        else:
            TP_,FP_,FN_, TN_ = kappa_eval(ref_mask, mask3, iou_thr, metric)
            cm_overlay = overlay(img, ref_mask, mask3)
        
        cv2.putText(cm_overlay, "TP: {}, FP: {}, FN: {}, TN: {}".format(TP_,FP_,FN_,TN_), 
                    (5,32), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
        io.imsave(os.path.join("./sample_out",f.split(os.sep)[-1]), cm_overlay)
        io.imsave(os.path.join("./ref_mask",f.split(os.sep)[-1][:-4]+'_ref_mask_contour.png'), ref_mask_overlay)
        
        TP += TP_
        FP += FP_
        FN += FN_
        TN += TN_
        
    csv.close()

    print("Total")
    print("TP: {}, FP: {}, FN: {}, TN: {}".format(TP,FP,FN,TN))
    print("*"*20)
    kappa = eval_metric(TP, FP, FN, TN)
    print("kappa: ",kappa)
    sensitivity = sensitivity_metric(TP, FP, FN, TN)
    print("Sensitivity: ",sensitivity)
    specificity = specificity_metric(TP, FP, FN, TN)
    print("Specificity: ",specificity)
    PPV = ppv_metric(TP, FP, FN, TN)
    print("Positive predictive value: ",PPV)
    NPV = npv_metric(TP, FP, FN, TN)
    print("Negative predictive value: ",NPV)      

#%%
'''
On running the STAPLE algorithmic code, we get the consensus reference annotation
which we use further to evaluate the performance of individual experts and the program. 
For instance, we use the STAPLE as the reference and compare its performance with
expert_1 for various IOU thresholds in the range (0.1 - 0.7) to measure 
kappa, sensitivity, specifcity, positive and negative predictive values.
We repeat it for every expert and the program ROI predictions to evaluate the performance.
'''
#%%
'''
Here, we show computing the ensemble localization using the top-3,
top-5 and top-7 models and measure their IOU and mAP scores.
The CRM activations are thresholded to remove values below 20% to alleviate
issues due to noise. We compute the mAP values for 10 different IOU thresholds
in the range (0.1 - 0.7).
'''
#%%
# create json format
def serializeGT(ImgName, g_coords):
    return {ImgName:  g_coords,}

def serializeCRM(ImgName, b_coords, b_scores):
    return {ImgName: { 
            'boxes': b_coords,
            'scores': b_scores,}
        }

#%%
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

#%%   
#we rename the layers of the VGG-16 MODEL to allevaite common layer names
#with VGG-19
 
def VGG16_DL(model_input, num_classes):
    try:
        vgg16_cnn = VGG16(weights='imagenet', include_top=False, input_tensor=model_input)
        vgg16_cnn=Model(inputs=vgg16_cnn.input,
                       outputs=vgg16_cnn.get_layer('block5_conv3').output)
        x = vgg16_cnn.output
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='extra_conv_vgg16')(x)             
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=vgg16_cnn.input, outputs=predictions, name='vgg16_custom')
        model.get_layer(name='block1_conv1').name='block1_conv1VGG'  
        model.get_layer(name='block1_conv2').name='block1_conv2VGG' 
        model.get_layer(name='block2_conv1').name='block2_conv1VGG' 
        model.get_layer(name='block2_conv2').name='block2_conv2VGG' 
        model.get_layer(name='block3_conv1').name='block3_conv1VGG' 
        model.get_layer(name='block3_conv2').name='block3_conv2VGG' 
        model.get_layer(name='block3_conv3').name='block3_conv3VGG' 
        model.get_layer(name='block4_conv1').name='block4_conv1VGG' 
        model.get_layer(name='block4_conv2').name='block4_conv2VGG' 
        model.get_layer(name='block4_conv3').name='block4_conv3VGG' 
        model.get_layer(name='block5_conv1').name='block5_conv1VGG' 
        model.get_layer(name='block5_conv2').name='block5_conv2VGG' 
        model.get_layer(name='block5_conv3').name='block5_conv3VGG' 
        model.get_layer(name='block1_pool').name='block1_poolVGG' 
        model.get_layer(name='block2_pool').name='block2_poolVGG' 
        model.get_layer(name='block3_pool').name='block3_poolVGG' 
        model.get_layer(name='block4_pool').name='block4_poolVGG' 
        model.get_layer(name='extra_conv_vgg16').name='extra_conv_VGG' 
        return model
    except Exception as e:
        raise Exception('Error from VGG16_DL(): ' + str(e))

#%%
def get_output_layer(model, layer_name):
    try:
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer
    except Exception as e:
        raise Exception('Error from get_output_layer(): ' + str(e))

#%%
def generate_BoundingBox(aCRM, threshold):
    try:
        labeled_CRM, nr_objects = ndimage.label(aCRM > threshold)
        props = regionprops(labeled_CRM)
        return props
    except Exception as e:
        raise Exception('Error from generate_BoundingBox(): ' + str(e))

#%%
# Confidence score = highest heatmap value in box * classification score
def Calculate_Confidence_Score(aCRM, bboxes, outScores):
    try:
        c_scores = []
        for a_b in bboxes:
            a_bbox = aCRM[a_b[1]:a_b[3], a_b[0]:a_b[2]]       
            a_score = np.max(a_bbox)
            b_score = outScores[0][0]
            c_scores.append(np.max(a_bbox) * outScores[0][0])
        
        return np.array(c_scores)
        
    except Exception as e:
        raise Exception('Error from Calculate_Confidence_Score(): ' + str(e))

#%%

def Generate_CRM_2Class(thisModel, thisImg_path, Threshold):   
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
                        tmp0[x,y] = 0.     # remove activation at the spatial location (x,y)
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
def generate_bBox(thisCAM_img, Threshold):             
    try:
        bboxes = []
        TheProps = generate_BoundingBox(thisCAM_img, Threshold)
        for b in TheProps:
            bbox = b.bbox
            xs = bbox[1]
            ys = bbox[0]
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]

            bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])         

        CRM_bboxes = np.vstack(bboxes)
        
        return CRM_bboxes

    except Exception as e:
        raise Exception('Error from generate_bBox(): ' + str(e))

#%%
def calculate_IOU (GT_bbox, a_bbox):
    try:
        iou = 0.
        for c_b in a_bbox:
            for g_b in GT_bbox:
                iou += bb_intersection_over_union(g_b, c_b)
        iou /= float(GT_bbox.shape[0])           # get average 

        return iou

    except Exception as e:
        raise Exception('Error from calculate_IOU(): ' + str(e))

#%%
#begin ensemble localization
#we repeat this to measure the localization performance 
#for single, top-3, top-5, and top-7 ensembles
        
def Load_Pretrained_Model(thisModel_path, thisModel_name):
    try:
        # load weights into new model
        loaded_model = load_model(os.path.join(thisModel_path, thisModel_name + '.h5'))
        return(loaded_model)
    except Exception as e:
        raise Exception('Error from Load_Pretrained_Model(): ' + str(e))
try:
    # Load Keras model
    print("Loading a total of 3 DL models...")
    try:
        img_width, img_height = 256, 256
        input_shape = (img_width, img_height, 3)
        model_input = Input(shape=input_shape)
        print(model_input)

        try:
            print("1) Loading VGG16-based model...")
            VGG16_M = VGG16_DL(model_input, 2)
            VGG16_M.load_weights(
                os.path.join('../weights/', 'vgg16_covid_finetuned_8020.01-0.8681' + '.h5'))
        except Exception as e:
            raise Exception('Error in loading VGG16-based model: ' + str(e))

        try:
            print("2) Loading VGG19-Based model...")
            VGG19_M = Load_Pretrained_Model('C:/Users/codes/weights', 
                                            'vgg19_covid_finetuned_8020.01-0.8611')
        except Exception as e:
            raise Exception('Error in loading VGG19-based model: ' + str(e))
            
        try:
            print("3) Loading InceptionV3-based model...")
            InceptionV3_M = Load_Pretrained_Model('C:/Users/codes/weights',
                                                  'iv3_covid_finetuned_8020.02-0.8611')
        except Exception as e:
            raise Exception('Error in loading InceptionV3-based model: ' + str(e))
        
        try:
            print("4) Loading ResNet18-based model...")
            ResNet18_M = Load_Pretrained_Model('C:/Users/codes/weights',
                                                  'resnet18_covid_finetuned_8020.02-0.8958')
        except Exception as e:
            raise Exception('Error in loading ResNet18-based model: ' + str(e))
        
        try:
            print("5) Loading MobileNet-V2-based model...")
            MobileNet_M = Load_Pretrained_Model('C:/Users/codes/weights',
                                                  'mobilev2_covid_finetuned_8020.03-0.8750')
        except Exception as e:
            raise Exception('Error in loading MobileNet-based model: ' + str(e))
        
        try:
            print("6) Loading DenseNet-121-based model...")
            DenseNet_M = Load_Pretrained_Model('C:/Users/codes/weights',
                                                  'densenet_covid_finetuned_8020.08-0.8750')
        except Exception as e:
            raise Exception('Error in loading DenseNet-121-based model: ' + str(e))
        
        try:
            print("7) Loading Xception-based model...")
            Xception_M = Load_Pretrained_Model('C:/Users/codes/weights',
                                                  'xception_covid_finetuned_8020.13-0.8681')
        except Exception as e:
            raise Exception('Error in loading Xception-based model: ' + str(e))

    except Exception as e:
        raise Exception('Error in loading CNN models: ' + str(e))
 
    try:
        TestFolder_path = 'expert1_annotations/' #images annotated by expert_1, change for different experts
        print("Input image folder: " + TestFolder_path)
        Test_image_path = TestFolder_path + '*.png'
        Test_image_Set = glob.glob(Test_image_path)
    except Exception as e:
        raise Exception('Error in retrieving images from dataset folder: ' + str(e))
    
    # Set-up output folders
    try:
        Visualization_path   = 'C:/Users/codes/visualization/' #create this folder to record the output

        aCRM_output_path      = Visualization_path + 'aCRM'       
        aCRM_bbox_output_path = Visualization_path + 'aCRM_bbox'        
        tCRM_output_path      = Visualization_path + 'tCRM'      
        tCRM_bbox_output_path = Visualization_path + 'tCRM_bbox'
        
        f1 = open(os.path.join(Visualization_path, 'y_ClassificationResult.txt'), 'w')

        print("CRM output folder: " + Visualization_path)
    except Exception as e:
        raise Exception('Error in setting up output folders: ' + str(e))

    # Load groundtruth information from csv file
    try:
        cf = open('expert_1_annotations.csv', 'r') #CSV file containing annotations from expert-1, replace csv with that generated for expert-2 and staple-generated consensus
    except Exception as e:
        raise Exception('Error in loading csv file: ' + str(e))

    Th_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] #setting different thresholds for CRM

    avg_aIOU = np.zeros(len(Th_set))       
    avg_tIOU = np.zeros(len(Th_set))       

    bb_data = [dict() for x in range(len(Th_set))]
    tb_data = [dict() for x in range(len(Th_set))]
    gt_data = [dict() for x in range(len(Th_set))]
    # #while using STAPLE-generated consensus ROI, we record the predictions of the program into a CSV file
    # csv_file = open('bounding_box_pred_program.csv','w')
    # csv_file.write("patientId,x_dis,y_dis,width_dis,height_dis,Labels\n")
    # print('writing the bounding box coordinates to a CSV file')

    i = 0
    err_cnt = 0
    for Img_filename in Test_image_Set:

        fname = os.path.basename(Img_filename)
        name_only, ext_only = os.path.splitext(fname)

        aCRM_ImgFile    = name_only           # for saving CRM heatmap image
        aBbox_file      = name_only           # for saving bbox image

        tCRM_ImgFile    = name_only           # for saving tCRM heatmap image
        tBbox_file      = name_only           # for saving bbox image

        # Check if a given test image has a ground-truth information (bbox) 
        #in the csv file. if not, skip the remaining process
        bbox_set = []
        FileFound = False
        BBoxFound = False
        k = 0
        cf.seek(0)
        CSVreader = csv.DictReader(cf)        
        for row in CSVreader:           # 'for loop' for dealing with the case that a given image has multiple bounding boxes (actually very common)
            if fname in row['patientId']:
                FileFound = True        # OK found image file name in the CSV, next check if it has ground-truth bbox info.

                if row['x_dis'] and row['y_dis'] and row['width_dis'] and row['height_dis']:
                    bbox_set.append([int(row['x_dis']), int(row['y_dis']), 
                                     int(row['x_dis'])+int(row['width_dis']), 
                                     int(row['y_dis'])+int(row['height_dis'])])
                    BBoxFound = True
                    continue            # continue 'for loop' to see if this image has multiple bboxes
                else:
                    break
            k += 1

                                  
        if FileFound:
            if BBoxFound:
                print('[' + str(i) + '] ' + fname + ': Found from ' + \
                        str(k) + 'th row in the dictionary')
            else:
                print('[' + str(i) + '] ' + fname + ': Found from ' + \
                        str(k) + 'th row in the dictionary (BBox info missing)')
                continue                                # we only care about the image having groundtrugh bbox information
        else:
            print('[' + str(i) + '] ' + fname + ': Not Found from the dictionary')
            continue

        groundtruth_bbox_set = np.vstack(bbox_set)      # a list of ground-truth bbox information
        
        #load the top-7 best performing model with respect to each radiologist annotation;
        # and STAPLE-generated consensus ROI. Run for individual models and ensemble
        # Here, we show how to do for the top-7 ensemble
        InImage1, OutScores1, aCRM_Img1, tCRM_Img1 = Generate_CRM_2Class(ResNet18_M,,Img_filename, 0.2) 
        InImage2, OutScores2, aCRM_Img2, tCRM_Img2 = Generate_CRM_2Class(MobileNet_M,Img_filename, 0.2) 
        InImage3, OutScores3, aCRM_Img3, tCRM_Img3 = Generate_CRM_2Class(DenseNet_M,Img_filename, 0.2)
        InImage4, OutScores4, aCRM_Img4, tCRM_Img4 = Generate_CRM_2Class(VGG16_M,Img_filename, 0.2)
        InImage5, OutScores5, aCRM_Img5, tCRM_Img5 = Generate_CRM_2Class(Xception_M,Img_filename, 0.2)
        InImage6, OutScores6, aCRM_Img6, tCRM_Img6 = Generate_CRM_2Class(VGG19_M, Img_filename, 0.2)
        InImage7, OutScores7, aCRM_Img7, tCRM_Img7 = Generate_CRM_2Class(InceptionV3_M, Img_filename, 0.2)
        
        # # to evalaute a single model
        # OutScores_F = OutScores1/1.0 #score normalization
        # #for top-3 ensemble
        # OutScores_F = (OutScores1 + OutScores2 + OutScores3)/3.0 #score normalization
        # #for top-5 ensemble
        # # OutScores_F = (OutScores1 + OutScores2 + OutScores3 + \
        # #               OutScores4 + OutScores5)/5.0 #score normalization
        #for top-7 ensemble
        OutScores_F = (OutScores1 + OutScores2 + OutScores3 + OutScores4 + \
                      OutScores5 + OutScores6 + OutScores7)/7.0 #score normalization
        eval_result = "{}:    [{:.3f},  {:.3f}]".format(fname, OutScores_F[0][0], OutScores_F[0][1])

        target = OutScores_F[0][0]
        TP = True                                                                                                                            
        for other_output in OutScores_F[0]:
            if target < other_output:
                TP = False
                break
                    
        if TP:          
            aCRM_ImgFile += "_aCRM.png"
            aBbox_file += "_aBbox.png"

            tCRM_ImgFile += "_tCRM.png"
            tBbox_file  += "_tBbox.png"
            eval_result += "      | "
        else:
            err_cnt += 1

            aCRM_ImgFile += "_aCRM_err.png"
            aBbox_file += "_aBbox_err.png"

            tCRM_ImgFile += "_tCRM_err.png"
            tBbox_file  += "_tBbox_err.png"
            eval_result += "  ERR\n"           

        # CRM image (does NOT apply thresholding to each individual CRM image)
        # aCRM_Img_F = aCRM_Img1 # for singel model
        # aCRM_Img_F = aCRM_Img1 + aCRM_Img2 + aCRM_Img3  # for top-3
        # aCRM_Img_F = aCRM_Img1 + aCRM_Img2 + aCRM_Img3 + aCRM_Img4 + aCRM_Img5# for top-5
        aCRM_Img_F = aCRM_Img1 + aCRM_Img2 + aCRM_Img3 + aCRM_Img4 + \
        aCRM_Img5 + aCRM_Img6 + aCRM_Img7 # for top-7
        aCRM_Img_F /= np.max(aCRM_Img_F)
        
        # CRM image (remove the area lower than a given threshold)
        # tCRM_Img_F = tCRM_Img1 #1
        # tCRM_Img_F = tCRM_Img1 + tCRM_Img2 + tCRM_Img3 # for 3
        # tCRM_Img_F = tCRM_Img1 + tCRM_Img2 + tCRM_Img3 + tCRM_Img4 + tCRM_Img5 # for 5
        tCRM_Img_F = tCRM_Img1 + tCRM_Img2 + tCRM_Img3 + tCRM_Img4 + \
        tCRM_Img5 + tCRM_Img6 + tCRM_Img7 # for 7
        tCRM_Img_F /= np.max(tCRM_Img_F)
        
        
        for idx in range(len(Th_set)):
        
            aBBox_coord = generate_bBox(aCRM_Img_F, Th_set[idx])
            aBBox_score = Calculate_Confidence_Score(aCRM_Img_F, aBBox_coord, OutScores_F)

            aHeatmap = cv2.applyColorMap(np.uint8(255*aCRM_Img_F), cv2.COLORMAP_JET)
            aHeatmap[np.where(aCRM_Img_F < Th_set[idx])] = 0            
            aImg = np.float32(aHeatmap) + np.float32(InImage1)          
            aImg = 255 * aImg / np.max(aImg)


            tBBox_coord = generate_bBox(tCRM_Img_F, Th_set[idx])
            tBBox_score = Calculate_Confidence_Score(tCRM_Img_F, tBBox_coord, OutScores_F)

            tHeatmap = cv2.applyColorMap(np.uint8(255*tCRM_Img_F), cv2.COLORMAP_JET)
            tHeatmap[np.where(tCRM_Img_F < Th_set[idx])] = 0            
            tImg = np.float32(tHeatmap) + np.float32(InImage1)          
            tImg = 255 * tImg / np.max(tImg)

           
            a_path      = aCRM_output_path + str(Th_set[idx]) + '/'
            if not os.path.exists(a_path):
                os.makedirs(a_path)              
            cv2.imwrite(os.path.join(a_path, aCRM_ImgFile), aImg)

            t_path      = tCRM_output_path + str(Th_set[idx]) + '/'
            if not os.path.exists(t_path):
                os.makedirs(t_path)              
            cv2.imwrite(os.path.join(t_path, tCRM_ImgFile), tImg)

            aIOU = calculate_IOU (groundtruth_bbox_set, aBBox_coord)
            tIOU = calculate_IOU (groundtruth_bbox_set, tBBox_coord)
            eval_result += "aIOU_" + str(Th_set[idx]) + ":  {:.4f} | ".format (aIOU)
            eval_result += "tIOU_" + str(Th_set[idx]) + ":  {:.4f} | ".format (tIOU)
 
            avg_aIOU[idx] += aIOU        
            avg_tIOU[idx] += tIOU        

            aOutImage = np.copy(aImg)
            for c_b in aBBox_coord:
                cv2.rectangle(aOutImage, (c_b[0], c_b[1]), (c_b[2], c_b[3]), (255,0,0), 2)
            for g_b in groundtruth_bbox_set:
                cv2.rectangle(aOutImage, (g_b[0], g_b[1]), (g_b[2], g_b[3]), (0,255,0), 2)

            ab_path      = aCRM_bbox_output_path + str(Th_set[idx]) + '/'
            if not os.path.exists(ab_path):
                os.makedirs(ab_path)              
            cv2.putText(aOutImage, "IoU: {:.4f}".format(aIOU), (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)    
            cv2.imwrite(os.path.join(ab_path, aBbox_file), aOutImage)

            tOutImage = np.copy(tImg)
            for c_b in tBBox_coord:
                cv2.rectangle(tOutImage, (c_b[0], c_b[1]), (c_b[2], c_b[3]), (255,0,0), 2)
                # # save coordinates to csv when using STAPLE-generated consensusr ROI
                # csv_file.write(fname+','+str(c_b[0])+','+str(c_b[1])+','+str(abs(c_b[0]-c_b[2]))+','+str(abs(c_b[1]-c_b[3]))+','+'COVID-19'+'\n')
                
            for g_b in groundtruth_bbox_set:
                cv2.rectangle(tOutImage, (g_b[0], g_b[1]), (g_b[2], g_b[3]), (0,255,0), 2)

            tb_path      = tCRM_bbox_output_path + str(Th_set[idx]) + '/'
            if not os.path.exists(tb_path):
                os.makedirs(tb_path)              
            cv2.putText(tOutImage, "IoU: {:.4f}".format(tIOU), (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)    
            cv2.imwrite(os.path.join(tb_path, tBbox_file), tOutImage)
        
            b  = serializeCRM(fname, aBBox_coord.tolist(), aBBox_score.tolist())
            tb = serializeCRM(fname, tBBox_coord.tolist(), tBBox_score.tolist())
            g  = serializeGT(fname, groundtruth_bbox_set.tolist())           
            
            if len(bb_data[idx]) > 0:
                bb_data[idx].update(b)
            else:
                bb_data[idx] = dict(b)           # create dictionary

            if len(tb_data[idx]) > 0:
                tb_data[idx].update(tb)
            else:
                tb_data[idx] = dict(tb)           # create dictionary

            if len(gt_data[idx]) > 0:
                gt_data[idx].update(g)
            else:
                gt_data[idx] = dict(g)           # create dictionary
     
        f1.write(eval_result + '\n')

        i += 1
        print (i)

    for Th in range(len(Th_set)):
        with open(os.path.join(Visualization_path, 'Ensemble7_BBox_predicted_' + str(Th_set[Th]) + '.json'), 'w') as outfile:
            json.dump(bb_data[Th], outfile)
        with open(os.path.join(Visualization_path, 'Ensemble7_tBBox_predicted_' + str(Th_set[Th]) + '.json'), 'w') as outfile:
            json.dump(tb_data[Th], outfile)
        with open(os.path.join(Visualization_path, 'Ensemble7_BBox_groundtruth_' + str(Th_set[Th]) + '.json'), 'w') as outfile:
            json.dump(gt_data[Th], outfile)


    avg_aIOU /= float(i-err_cnt)
    avg_tIOU /= float(i-err_cnt)
    f1.write("\n\nTotal Errors: {:d}\n".format(err_cnt))
    f1.write("No. of testing images (having groundtruth bbox): {:d}\n".format(i))
    f1.write("Accuracy: {:.5f}\n".format((float(i-err_cnt)/float(i))*100.))

    for ix in range(len(avg_aIOU)):
        f1.write("avrage aIOU_" + str(Th_set[ix]) + ": {:.4f}  |".format(avg_aIOU[ix]))                                            
        f1.write("avrage tIOU_" + str(Th_set[ix]) + ": {:.4f}\n".format(avg_tIOU[ix]))                                            
    f1.close()
     
    print('done')
    csv_file.close()

except Exception as e:
    print (str(e))

#%%
'''
The results will be stored to the classificationresult.txt file.
Use the corresponding ensemble json file for the ground truth and predicted bounding boxes
toward measuring the MAP values as shown below
'''
#%%
sns.set_style('white')
sns.set_context('poster')

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    
#%%

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

#%%
def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`) and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx    = []
        pred_match_idx  = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes)   - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

#%%
def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos  += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

#%%
def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

#%%
def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5): #anything above 0.5 is good, but we use 0.2
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given model_threshold
            'recall' (list of floats): recall value for given model_threshold
            'models_thrs' (list of floats): model threshold value that precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes']  = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions  = []
    recalls     = []
    model_thrs  = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes']  = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}

#%%
def plot_pr_curve(precisions, recalls, category='Person', label=None, color=None, ax=None):    
 
    if ax is None:
        plt.figure(figsize=(20,10), dpi=300) 
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    
    #ax.scatter(recalls, precisions, label=label, s=10, color=color)
    ax.plot(recalls, precisions, label=label)
    ax.set_xlabel('recall', fontsize='medium')
    ax.set_ylabel('precision', fontsize='medium')
    ax.set_title('Precision-Recall curve\n')
    ax.set_xlim([0,1.1]) 
    ax.set_ylim([0,1.1])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 0.20))
    ax.tick_params(axis='x',which='major',direction='out',length=8,pad=-5,labelsize=20)
    ax.tick_params(axis='y',which='major',direction='out',length=8,pad=-5,labelsize=20)
    return ax

#%%

if __name__ == "__main__":

    bBox_groundtruth_File = 'C:/Users/codes/visualization/Ensemble7_BBox_groundtruth_0.7.json'
    bBox_prediction_File = 'C:/Users/codes/visualization/Ensemble7_tBBox_predicted_0.7.json'
    print("\n ****** Calculating mAP:  " + bBox_prediction_File + "****** \n")
    
    with open(bBox_groundtruth_File) as infile:
        gt_boxes = json.load(infile)

    with open(bBox_prediction_File) as infile:
        pred_boxes = json.load(infile)

    # Runs it for one IoU threshold
    iou_thr = 0.1 
    start_time = time.time()
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    #MAP values measured for 10 different IOU thresholds in the range (0.1 - 0.7)
    for idx, iou_thr in enumerate(np.linspace(0.1, 0.7, 10)):         
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']
        ax = plot_pr_curve(
            precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

   avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('mAP: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)    
    legend = plt.legend(loc='upper right', title='IOU Thr', fontsize='x-small', frameon=True)
    legend.get_title().set_fontsize('20')
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.show()

#%%
'''
The final results will show the PR curves and the 
MAP values measured for 10 different IOU thresholds in the range (0.1 - 0.7)
'''
