# Interpreting Deep Ensemble Learning through Radiologist Annotations for COVID-19 Detection in Chest Radiographs

Data-drive deep learning methods using convolutional neural networks demonstrate promising performance in natural image computer vision tasks. However, using these models for medical computer vision tasks suffer from the following limitations: (i) Difficulties in extending and optimizing their use since medical images have unique visual characteristics and properties unlike natural images; (ii) stochastic optimization and backpropagation-based learning strategy that models random noise during training, leading to variance errors, and poor generalization to real-world medical data; (iii) Lack of statistical analyses that provide reliable measures on performance variability and their effects on arriving at accurate inferences; (iv) black-box behavior that prevents explaining the learned interpretations, which is a serious bottleneck in deploying them for medical screening/diagnosis; and (v) complications in obtaining annotations and analyzing inter-reader variability that may lead to a false diagnosis or inability to evaluate the true benefit of accurately supplementing clinical-decision making. 

In this study, we propose a stage-wise, planned approach to address these limitations toward COVID-19 detection using chest X-rays, as follows: (i) we propose the benefits of repeated CXR-specific pretraining in transferring and fine-tuning the learned knowledge toward improving COVID-19 detection performance; (ii) we construct ensembles of the fine-tuned models to improve performance compared to individual constituent models; (iii) Statistical analyses is performed at various learning stages while reporting results and evaluating claims using quantitative measures; (iv) The learned behavior of the individual models and their ensembles are interpreted through class-selective relevance mapping-based region of interest localization that identifies discriminative ROIs involved in decision making; (v) We use the annotations of two expert radiologists, analyze inter-reader variability, and ensemble localization performance using Simultaneous Truth and Performance Level Estimation methods and investigate for the existence of statistically significant differences in Intersection over Union and mean average precision scores. 

We observe the following: (i) Ensemble approach improved classification and localization performance; (ii) Inter-reader variability and performance level assessment indicate the need to modify the algorithm and/or its parameters toward improving classification and localization. To our best knowledge, this is the first study to construct ensembles, perform ensemble-based disease ROI localization, and analyze inter-reader variability and algorithm performance, toward COVID-19 detection in CXRs.  

## Code description

This repository has two python codes and one R code: 

The code ensemble_learning.py is used as the code base for the following: i) UNet based semantic semgnetation to create lung masks for the datasets used in this study; (ii) perform repeated CXR-specific pretraining; (ii) Fine-tuning on COVID-19 detection; (iv) create ensembles of fine-tuned models to improve performance.

The code ensemble_visualization_inter_reader-variability_analysis.py is used as the code base for the following: (i) perform localization studies using ensemble CRM; (ii) compute PR curves for the model versus radiologists and model versus staple generated consensus annotation; (iii) analyse inter-reader variability using kappa and other measures. The code anova_code. R is a R code that shows the steps involved in performing statistical analyses including one-way ANOVA, Shaipiro-Wilk, and Levene's tests for this study. 

## Codes and Model weights

The following weights are added to the https://drive.google.com/drive/folders/1H4cnKXhNoC5mD6GjKCHhFRAFjezQHQeT?usp=sharing.

vgg19_modality_specific-0.6913.h5 - best performing, CXR modality specific VGG19 model.

resnet18_modality_specific-0.6821.h5 - CXR modality specific Resnet-18 model.

resnet18_multiclass-0.8936.h5 - CXR modality specific Resnet-18 model retrained for multi-class classification (normal, bacterial, non COVID-19 viral).

densenet_multiclass-0.9177.h5 - best performing, multiclass model (normal, bacterial, non COVID-19 viral).

resnet18_covid_finetuned-0.8958.h5 - best performing, resnet multiclass model finetuned for COVID-19 detection. 

The details about the models, the training process, are discussed in the publication available at https://www.medrxiv.org/content/10.1101/2020.07.15.20154385v1.full.pdf+html

You can feel free to use the vgg19_modality_specific-0.6913.h5, best performing, CXR modality specific VGG19 model, truncate and add task specific layers for your custom classificstion task, since the model weight layers are modality-specific ( trained on more than 160,000 CXRs to broadly learn the characteristics of normal and abnormal lungs). Look into the publication at https://www.medrxiv.org/content/10.1101/2020.07.15.20154385v1.full.pdf+html for more details.

### Using the trained models

First, use the unet.hdf5 weights file in the shared google drive link to generate lung masks of 256×256 pixel resolution. Use the following code snippet to crop the lung boundaries using the generated lung masks and store them as a bounding box containing all the lung pixels, as shown below:

```
model = load_model("C:/Users/trained_model/unet.hdf5")
model.summary()
test_path = "C:/Users/data/test" #where your test data resides
save_path = "C:/Users/data/membrane/result" #where you want the generated lung masks to be stored

data_gen_args = dict(rotation_range=10.,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=5,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest') 

testGene = testGenerator(test_path)
results = model.predict_generator(testGene,steps per epoch=135,verbose=1, workers=1, use_multiprocessing=False) #steps per epoch is the no. of samples in test image.
saveResult(save_path, results, test_path)

#custom function to generate bounding boxes
def generate_bounding_box(image_dir: str, #containing images
                          mask_dir: str, #containing masks, images have same name as original images
                          dest_csv: str, #CSV file to write the bounding box coordinates
                          crop_save_dir: str): #save the cropped bounding box images
    """
    the orginal images are resized to 256 x 256
    the output crops are resized to 256 x 256
    """
    if not os.path.isdir(mask_dir):
        raise ValueError("mask_dir not existed")

    case_list = [f for f in os.listdir(mask_dir) if f.split(".")[-1] == 'png'] #all mask images are png files

    with open(dest_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        for j, case_name in enumerate(case_list):
            mask = cv2.imread(mask_dir + case_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = cv2.imread(image_dir + case_name, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256)) #original images are resized to 256 x 256
            if mask is None or image is None:
                raise ValueError("The image can not be read: " + case_name)

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
            
            # new coordinates for image which is 1 times of mask, mask images are 256 x 256, 
            #so need to multiply 1 times to get 256 x 256, and relaxing the borders by 5% on all directions
            up_down_loose = int(1 * (down - up + 1) * 0.05)
            image_up = 1 * up - up_down_loose
            if image_up < 0:
                image_up = 0
            image_down = 1*(down+1)+up_down_loose
            if image_down > image.shape[0] + 1:
                image_down = image.shape[0]

            left_right_loose = int(1 * (right - left) * 0.05)
            image_left = 1 * left - left_right_loose
            if image_left < 0:
                image_left = 0
            image_right = 1*(right + 1)+left_right_loose
            if image_right > image.shape[1] + 1:
                image_right = image.shape[1]

            crop = image[image_up: image_down, image_left: image_right]
            crop = cv2.resize(crop, (256, 256)) #the cropped image is resized to 256 x 256

            cv2.imwrite(crop_save_dir + case_name, crop) # cropped images saved to crop directory

            # write new csv
            crop_width = image_right - image_left + 1
            crop_height = image_down - image_up + 1

            csv_writer.writerow([case_name,
                                 image_left,
                                 image_up,
                                 crop_width,
                                 crop_height]) #writes xmin, ymin, width, and height

            if j % 50 == 0:
                print(j, " images are processed!")

generate_bounding_box("C:/Users/data/test/",
                      "C:/Users/result/mask/",
                      'C:/Users/result/bounding_box.csv',
                      "C:/Users/result/cropped/")
```
Use the generated lung crops for your test data and then used the model weights available in the shared google link for your purpose. For instance, if you want to classify the CXRs as showing normal or abnormal lungs, use vgg19_modality_specific-0.6913.h5 that delivered the best performance in this task. If you want to perform a multi-class classification (normal, bacterial, non COVID-19 viral), use the densenet_multiclass-0.9177.h5 that delivered the best performance in this task. If you want to classify normal and COVID-19+ CXRs, use the resnet18_covid_finetuned-0.8958.h5 model weights toward this task. Gve the path to your directory where you have stored the images to be predicted. 

The data folder is organized as follows (depending on the task under study):
data 
|-train
  |-normal
  |-bacterial
  |-non COVID-19 viral
  |-COVID-19 viral
|-test
  |-normal
  |-bacterial
  |-non COVID-19 viral
  |-COVID-19 viral
  
 We used Keras ImageDataGenerator to preprocess the test images as follows:
 
 ```
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

nb_test_samples = len(test_generator.filenames)
print(test_generator.class_indices)
#true labels
Y_test=test_generator.classes
print(Y_test.shape)
#convert test labels to categorical
Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)
```
Predict on the test images as follows:

```
model = load_model('resnet18_covid_finetuned-0.8958.h5') 
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
test_generator.reset()
y_pred = model.predict_generator(test_generator, 
                                 nb_test_samples/batch_size, workers=1)
#true labels
Y_test=test_generator.classes
#print the shape of y_pred and Y_test
print(y_pred.shape)
print(Y_test.shape)
model_accuracy=accuracy_score(Y_test,y_pred.argmax(axis=-1))
print('The accuracy of model is: ', model_accuracy)
```

### Generate LIME-based decisions

To visualize the learned behavior of the pruned models, we shown an instance of how to use LIME visualization with the trained models:

```
model = load_model('resnet18_covid_finetuned-0.8958.h5') 
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#path to image to visualize
img_path = 'image1.png'
img = image.load_img(img_path)
#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255 
#predict on the image
preds = model.predict(x)[0]
print(preds)
#initialize the explainer
from lime import lime_image
from skimage.segmentation import mark_boundaries
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(x[0], 
                                         model.predict, top_labels=1, 
                                         hide_color=0, num_samples=42)
print(explanation.top_labels[0])
temp, mask = explanation.get_image_and_mask(0, #change the respective class index
                                            positive_only=False, 
                                            num_features=5, hide_rest=False) 
plt.figure()
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.figure()
plt.imshow(x[0] / 2 + 0.5) #this increases te brightness of the image

```

## Generate CAM based ROI localization

Visualize the learned behavior using CAM-based ROI localization
```
model = load_model('resnet18_covid_finetuned-0.8958.h5') 
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#path to image to visualize
img_path = 'image1.png'
img = image.load_img(img_path)
#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255 
#predict on the image
preds = model.predict(x)[0]
print(preds)
#begin visualization
covid_output = model.output[:, 0] 
#Output feature map from the deepest convolutional layer
last_conv_layer = model.get_layer('extra_conv_resnet18')
#compute the Gradient of the expected class with regard to the output feature map of the deepest convolutional layer)
grads = K.gradients(covid_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(1024): #number of filters in the deepest convolutional layer
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#For visualization purposes, we normalize the heatmap between 0 and 1.
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
img = cv2.imread(img_path) #path to the image
#Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#Converts the heatmap to RGB 
heatmap = np.uint8(255 * heatmap)
#Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img # 0.4 here is a heatmap intensity factor.
#Saves the image to disk
cv2.imwrite('cam_image.png', superimposed_img)

```
