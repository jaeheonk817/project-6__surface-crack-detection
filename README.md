![image](https://user-images.githubusercontent.com/122312679/232519248-bc3bde19-1dfe-4f4d-8d3d-1103ee86c9c2.png)

# Surface Crack Detection using Deep Learning Image Classification
authored by Jae Heon Kim

## Overview
This is a deep learning project that trains a convolutional neural network model to distinguish wall as either cracked or not-cracked based on image input. Some models include transfer learning from VGG16 or Resnet50. To view the project, please refer to `main_notebook.ipynb`. `data_prework.ipynb` file contains tedious exploratory data anaysis and data organization work not crucially important in understanding overall flow of the project but I kept it there for reference. Also, `deployment.ipynb` is was created solely for streamlit deployment.

## Business Understanding
The stakeholder for this CRISP-DM data science project is the NYC Department of Buildings. The problem they face is that current inspection methods are outdated, costly, time-consuming, and less accurate due to human imperfections. This means that required inspections are not being completed on time, and there is a need for a more efficient and accurate solution. The stakeholder is interested in a model that can help digitize this inspection process, particularly with detecting building cracks. The goal of this project is to train a convolutional neural network (CNN) model that can accurately classify walls as cracked or not-cracked based on image input, thus providing a more automated and accurate solution for building inspections. By achieving this goal, the NYC Department of Buildings can improve the accuracy and efficiency of their inspection process, ultimately leading to cost savings and more timely inspections, which in turn can benefit and protect everyone living and working in New York City.

## Data Understanding
The dataset consists of 40,000 images of concrete surfaces with and without cracks, divided into two separate folders for negative (without crack) and positive (with crack) image classification. Each class has 20,000 images with a resolution of 227 x 227 pixels with RGB channels. The data is generated from 458 high-resolution images (4032x3024 pixel) using the method proposed by Zhang et al. (2016). No data augmentation in terms of random rotation or flipping or tilting is applied. The dataset is contributed by Çağlar Fırat Özgenel and was previously used in a study comparing the performance of pre-trained Convolutional Neural Networks (CNNs) on crack detection in buildings (Özgenel & Gönenç Sorguç, 2018). The dataset can be used for classifying crack and non-crack images and for creating a mask of the cracked portion in new images using image segmentation methods.

Source: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

Following are randomly pulled samples from the data:

![image](https://user-images.githubusercontent.com/122312679/232523263-687bff2c-b827-4e49-89fa-c52aa4e7b5ca.png)
![image](https://user-images.githubusercontent.com/122312679/232523295-20dadc2c-0ad7-4127-bf2e-23845ffac183.png)

## Data Preparation

The original data was located in a single directory with two categories: positive and negative. After reviewing the data, the data was randomly split into 80% for training, 10% for validation, and 10% for testing using sklearn's train test split. Substantial portion of data was allocated to the training set due to the data's richness, which still allows for strong validation and testing sets even at this ratio. The process is documented in the `data_prework.ipynb`.

All images had their pixel values normalized(scaled) and they were data-augmented to help model's robustness. Data augmentation techniques used are zooming, flipping, rotating, grayscaling and shifting.

## Modeling
The overall accuracy of a model is undoubtedly crucial, but it's important to consider whether false positives or false negatives are more concerning. False positives may result in a false alarm that triggers a careful survey by human workers at the scene. On the other hand, false negatives can be more problematic as they can endanger individuals who may be inside or near a building with structural damages. Therefore, while the model's goal should be to achieve perfect accuracy, it should prioritize minimizing false negative errors before minimizing false positives.

### First Model (Baseline)
![image](https://user-images.githubusercontent.com/122312679/232525164-7183ac97-6f76-485f-965e-c043950c5761.png)


## Evaluation
### Final Model's Errrors
It's important to individually review the final model's errors to see if there is any pattern. Also this can lead to unexpected insights for improving the final model.

![image](https://user-images.githubusercontent.com/122312679/232531778-83b493a9-e7d6-4938-8c49-2811c295c39c.png)
- `False Positive 1` has a line shape in upper left that could have been mistaken for a crack.
- `False Positive 2` has no clear reason as to why it can be mistaken for a cracked wall.
- `False Positive 3` has a linear pattern that can be mistaken for a crack which to humans eys shouldn't be mistaken.
- `False Positive 4` seems like the image actually has a crack but mislabled by humans compiling the data.
![image](https://user-images.githubusercontent.com/122312679/232532046-d196e2f7-60d7-4a6d-9ffa-5a3ccca58417.png)
- `False Negative 1` has a slight divergence visible to naked human eyes.
- `False Negative 3` seemingly has no crack, or maybe a very slight one. This is hard to notice.
- `False Negatives 2, 4, 5` all have crackes on the edge of their images. The model needs to be trained on such occasions since 60% of false negative errors are this type.

