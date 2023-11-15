# Image-classification-on-CIFAR-10-dataset-using-CNN
Built a CNN model for classifying images on CIFAR-10 dataset. Tried with different CNN architectures and applied different regularization techniques such as Dropout, Batch-normalization and data augmentation and observed the improvement in the performance of the model

Requirements: Tensorflow, Keras, Matplotlib, Seaborn\
Performance metrics: Accuracy

### About the dataset:
Image dimensions: 32x32x3\
Total no. of images = 60,000\
No. of train images = 50,000\
No. of test images = 10,000\
Total number of classes = 10\
No. of train images (classwise) = 5,000\
No. of test images (classwise) = 1,000.


### Performance of the model with different architectures and different regularization techniques:


| S.No | CNN Architecture        | No. of iterations | Test accuracy  |
| ---- |:-----------------------:| -----------------:| --------------:|
| 1    | With 1 VGG blocks        | 20             |       67.81%         |
| 2    | With 2 VGG blocks              | 20             |     72.85%           |
| 3    | With 3 VGG blocks                | 20             |      72.89%          |
| 4    | With 3 VGG blocks + dropout(0.2)        | 50             |   79.6%             |
| 5    | With 3 VGG blocks + dropout(0.3)              | 50             |   79.17%             |
| 6    | With 3 VGG blocks + dropout(0.4)                | 50             |   76.84%             |
| 7    | With 3 VGG blocks + dropout(0.3) + batch normalization              | 50             |   86.12%             |
| 8    | With 3 VGG blocks + dropout(0.3) + batch normalization + data augmentation              | 50             |   88.27%             | 
