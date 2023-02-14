# Traffic Sign Classification 🚦
Traffic Sign Classification using Convolutional Neural Network (CNN). The [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) dataset, which contains thousands of images from 43 classes of signs, is used for training and testing. 

This project is a part of CS50AI course. Click [here](https://cs50.harvard.edu/ai/2020/projects/5/traffic/) for further project details.

## Execution
Download [GTSRB  dataset](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip) and save as `gtsrb` to the same directory with `traffic.py`. Then, run
```
python traffic.py gtsrb
```
or
```
python traffic.py gtsrb [model_name]
```
to save the output model to the directory as `model_name`.

## Experimentation Process
### 1. Initiate a baseline model
One convolutional layer, one max pooling layer with size 2x2, one hidden layer, and one dropout layer
- Convolutional layer: Learn 32 filters using a 3x3 kernel
- Max-pooling layer: size 2x2
- Flatten unit
- Hidden layer: dim 128
- Dropout layer: rate = 0.5
- Output layer

Accuracy: 5.69 % 💀

### 2. Varying dropout layers and dropout rate
Changing dropout rate from 0.5 to 0.2 improved accuracy from 5% to 86% 📈.
This suggests the significance of dropout rate. Also, adding dropout layer after each convolutional and hidden layer helps preventing overfitting.

### 3. Varying number of convolutional layers and kernel size
Adding more convolutional layers and increasing filter sizes makes the model improved to over 90% accuracy 📈. 
However, three-layers case was worse which can be because at that point the input size is already too small for another convolution.

### 4. Varying max pooling size
Adding up pooling size to over 2 didn't help improving accuracy in my case.

### 5. Varying number and size of hidden layers
Finally, in my case, adding up size of the hidden layer helps while adding more hidden layers doesn't improve the performance of the model.

### 6. Final model
Model structure
- Convolutional layer: 16 filters, 5x5 kernel
- Max-pooling layer: size 2x2
- Dropout layer: rate = 0.2
- Convolutional layer: 32 filters, 5x5 kernel
- Max-pooling layer: size 2x2
- Dropout layer: rate = 0.2
- Flatten unit
- Hidden layer: dim 512
- Dropout layer: rate = 0.2
- Output layer

Accuracy: 96.88 % 🤗
