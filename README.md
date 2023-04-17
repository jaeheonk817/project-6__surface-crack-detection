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
First model is a simple convolutional neural network with just a single hidden layer (flattening). Since this is a binary classification the output activation function was sigmoid, and will hold true for all subsequent models. The accuracy of this model on the test data was 0.929, and the recall was 0.894. Having an impressive metric like this out of such a simple model is an indicator that the study should aim at reaching near-perfect metrics when more complicated models are trained.

### Subsequent Models' Architecture
The following are subsequent models' archituectures:
- `Second Model`: This model a deep neural network with 4 layers, consisting of 1 convolutional layer with 32 filters, followed by 3 fully connected layers with 128, 64, and 1 neurons respectively. The model also includes batch normalization and dropout layers for regularization to improve accuracy and prevent overfitting. The total number of trainable parameters in the model is approximately 67 million.
- `Third Model`: Third model was is a much simpler version. It's has same architecture as the first model except it inputs (128, 128, 1) gray-scale images as opposed to color images. This model's purpose was to test whether training can be done much quicker without loss of accuracy.
- `Fourth Model`: This model loads the VGG16 pre-trained model without the top layer and freezes its convolutional base. Then, it defines a new neural network model with the VGG16 base as a layer, followed by a flatten layer to convert the output of the base into a 1D vector. The model then adds two fully connected dense layers with a dropout layer in between, and finally, a single output sigmoid activation layer. The model is compiled with binary cross-entropy loss, accuracy, and recall as metrics. Early stopping and learning rate scheduling callbacks are defined. Finally, the model is fine-tuned by fitting it on the training data for six epochs and validating on the validation data, using the early stopping and learning rate scheduling callbacks.
- `Fifth Model`: This model had 3 extra hidden fully-connected layers before the final output compared to the fourth model to retain more information.
- `Sixth Model`: This model is a replication of the fifth model except that it uses transfer learning from Resnet50 instead of VGG16.

### Subsequent Models' Metrics
![image](https://user-images.githubusercontent.com/122312679/232550303-41410d3d-2aae-4164-b680-467e07798fb1.png)
- The dotted vertical line is the average of all recall values.
- The dotted horizontal line is the average of all accuracy values.
- Fifth model is absolutely our top performer in all metrics. We will choose it as our final model for deployment.

## Evaluation
### Final Model's Errors on Test Data
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

### Deployment
During deployment of a computer vision application using Streamlit, the input images contained more complex backgrounds with additional objects such as murals, unlike the training images that only contained wall images.  Also, unlike training process, some wall images were made of bricks, which is something completely unseen by the model. Despite this difference in the input images, the deployed model was able to accurately predict the labels of all the images, including the ones with added noise and complexity in the background. The first three have cracks and the next three don't.

<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/122312679/232543486-9f4d2cab-2f40-4d86-9e49-6ce0ee42f259.jpeg" width="250px"></td>
    <td><img src="https://user-images.githubusercontent.com/122312679/232543591-333ff39f-cf0c-4b75-b76f-a1585e385aa8.jpeg" width="250px"></td>
    <td><img src="https://user-images.githubusercontent.com/122312679/232543615-02441cba-d1a1-4d69-b64e-6c84cfc1422b.jpeg" width="250px"></td>
  </tr>
</table>
Above three wall images were correctly identified as cracked wall images.

<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/122312679/232543631-6c42e75b-2d91-454c-8eb9-b1302cccd1d9.jpeg" width="250px"></td>
    <td><img src="https://user-images.githubusercontent.com/122312679/232543641-4c4f596e-cdb5-4012-9baa-3f627c8ae911.jpeg" width="250px"></td>
    <td><img src="https://user-images.githubusercontent.com/122312679/232543656-96577210-9350-476e-8116-b692a576bc3f.jpeg" width="250px"></td>
  </tr>
</table>
Above three wall images were correctly identified as non-cracked wall images.

## Conclusion
### Final Model's Performance
- The final model's accuracy of 99.8% is an increase of 6.9% from baseline model's accuracy of 92.9%.
- The final model's recall of 99.8% is an increase of 10.4% from baseline model's recall of 89.4%
- Some of wrongly predicted images are difficult for human eyes to judge as well.
- This model will allow for a near-perfect classification at a tiny fraction of time humans perform the job.

### Recommendations to NYC Department of Buildings
- Optimize human resources and cut costs by deploying less specialized personnel for inspection tasks that can be simplified through the use of this model. Rather than hiring expensive, over-qualified inspectors, consider deploying less expensive personnel to simply capture images for inspections. This approach will not only significantly cut costs, but also allow for more inspections that are currently behind schedule.
- Leverage the power of the New York City Department of Buildings to gather more data for building inspections. Since buildings are often private properties, it can be challenging for third-party inspection companies to obtain detailed data. However, by utilizing the lawful power granted to the NYC Department of Buildings, we can gather more 2D and 3D data about buildings, which can be used not only for building inspections, but also for a wide range of future applications that may not yet be fully realized.
- Use the crack detection model to gather time-series image data for early identification of potential cracks. By monitoring the evolving condition of building walls over time, the inspection company can identify walls that have a higher risk of developing cracks, and take proactive measures to prevent further damage. This can save the building owner significant costs and resources, and enhance the inspection company's reputation as a trusted and reliable service provider. Building a time-series model for building condition prediction can help judge a building not only on the face value of its current status but also on overall degrading pace over time and where it's going.

### Next Steps
- Collect more diverse data sets, including images taken from various distances and angles, with obstacles present in the background.
- Train the model on moving images (3D) to expedite the inspection process and provide inspectors with a point-of-view camera to capture images for analysis.
- Enhance the model's capabilities by expanding it to include the detection of other building features, such as rotting and rat holes.
- Test the model with a larger and more diverse set of data using the Streamlit platform to evaluate and refine its performance.
- Incorporate additional training data that includes images that were not previously seen to continue improving the model's accuracy.
- Create an ensemble model, such as a voting classifier, that combines multiple well-performing models of different types to account for the weaknesses of any single model and improve the accuracy and usefulness of the overall system.
