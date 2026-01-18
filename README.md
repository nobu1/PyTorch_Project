# PyTorch_Project  
This project demonstrates PyTorch implementation using Google Colab.  


## Features of PyTorch Compared to Keras  
### Pros
- Feature-rich and highly expandable  
- Easy to develop custom features as needed  

### Cons  
- The methods for defining machine learning models are somewhat complex  
- We need to write the training process for the machine learning model yourself  

## Execution environment  
- Google Colab

## Overview of PyTorch Implementation  
![PyTorchImplementation](./img/ImplementationMethodPyTorch.png)  
<br>

## Layer Functions  
- Functions that take a tensor as input and produce a tensor as output  

| Function | Parameters | Model Type |
|---|---|---|
| nn.Linear | **Required** | Regression |
| nn.Sigmoid | None | Binary Classification |
| nn.LogSoftmax | None | Multinominal Classification |
| nn.ReLU | None | Digit Recognition |
| nn.Conv2d | **Required** | Image Recognition |
| nn.MaxPool2d | None | Image Recognition |
| nn.Flatten | None | Image Recognition |
| nn.Dropout | None | Image Recognition |
| nn.BatchNorm2d | **Required** | Image Recognition |
| nn.AdaptiveAvgPool2d| None | Pre-trained Model |

## Linear Regression  
[Linear Regression Task](./LinearRegression/linear_regression.md)  

## Binary Classification  
[Binary Classification Task](./BinaryClassification/binary_classification.md)  

## Multinominal Classification  
[Multinominal Classification Task](./MultinomialClassification/multinomial_classification.md)  

## Digit Recognition using MNIST  
[Digit Recognition MNIST](./DigitRecognitionMNIST/digit_recognition_MNIST.md)  

## Image Recognition by CNN  
[Image Recognition by CNN](./ImageRecognitionCNN/image_recognition_CNN.md)  

## Tuning Techniques  
[Tuning Techniques Task](./TuningTechniques/tuning_techniques.md)  

## Using Pre-trained Model  
[Pre-trained Model Task](./Pre-trainedModel/pre-trained_model.md)  

## Custom Data Image Recognition  
[Custom Data Image Recognition Task](./CustomDataImageClassification/custom_data_image_classification.md)  
