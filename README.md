# Explainable-ensemble-learning-and-inter-reader-variability-analysis-for-COVID-19-detection

Data-drive deep learning methods using convolutional neural networks demonstrate promising performance in natural image computer vision tasks. However, using these models for medical computer vision tasks suffer from the following limitations: (i) Difficulties in extending and optimizing their use since medical images have unique visual characteristics and properties unlike natural images; (ii) stochastic optimization and backpropagation-based learning strategy that models random noise during training, leading to variance errors, and poor generalization to real-world medical data; (iii) Lack of statistical analyses that provide reliable measures on performance variability and their effects on arriving at accurate inferences; (iv) black-box behavior that prevents explaining the learned interpretations, which is a serious bottleneck in deploying them for medical screening/diagnosis; and (v) complications in obtaining annotations and analyzing inter-reader variability that may lead to a false diagnosis or inability to evaluate the true benefit of accurately supplementing clinical-decision making. 

In this study, we propose a stage-wise, planned approach to address these limitations toward COVID-19 detection using chest X-rays, as follows: (i) we propose the benefits of repeated CXR-specific pretraining in transferring and fine-tuning the learned knowledge toward improving COVID-19 detection performance; (ii) we construct ensembles of the fine-tuned models to improve performance compared to individual constituent models; (iii) Statistical analyses is performed at various learning stages while reporting results and evaluating claims using quantitative measures; (iv) The learned behavior of the individual models and their ensembles are interpreted through class-selective relevance mapping-based region of interest localization that identifies discriminative ROIs involved in decision making; (v) We use the annotations of two expert radiologists, analyze inter-reader variability, and ensemble localization performance using Simultaneous Truth and Performance Level Estimation methods and investigate for the existence of statistically significant differences in Intersection over Union and mean average precision scores. 

We observe the following: (i) Ensemble approach improved classification and localization performance; (ii) Inter-reader variability and performance level assessment indicate the need to modify the algorithm and/or its parameters toward improving classification and localization. To our best knowledge, this is the first study to construct ensembles, perform ensemble-based disease ROI localization, and analyze inter-reader variability and algorithm performance, toward COVID-19 detection in CXRs.  

## Code description

This repository has two python codes and one R code: 

The code ensemble_learning.py is used as the code base for the following: i) UNet based semantic semgnetation to create lung masks for the datasets used in this study; (ii) perform repeated CXR-specific pretraining; (ii) Fine-tuning on COVID-19 detection; (iv) create ensembles of fine-tuned models to improve performance.

The code ensemble_visualization_inter_reader-variability_analysis.py is used as the code base for the following: (i) perform localization studies using ensemble CRM; (ii) compute PR curves for the model versus radiologists and model versus staple generated consensus annotation; (iii) analyse inter-reader variability using kappa and other measures. 4

The code anova_code. R is a R code that shows the steps involved in performing statistical analyses including one-way ANOVA, Shaipiro-Wilk, and Levene's tests for this study. 
