## S3 Assignment


**Main Task**

Write a neural network that can:

    * take 2 inputs:
        * an image from MNIST dataset, and
        * a random number between 0 and 9  
        
    * and gives two outputs:
        * the "number" that was represented by the MNIST image, and
        * the "sum" of this number with the random number that was generated and sent as the input to the network



![](https://tenor.com/S7pU.gif)



###### Assignment Solution: ![S3-Solution](https://github.com/Gilf641/EVA-6/blob/main/Assignments/S3/EVA6_S3.ipynb)



### Steps:

* **Dataset Representation**

I have used idx3-ubyte format to store and save MNIST Dataset. It's loaded as an archive file(.gz format). 

* **Dataset Generation strategy**

For MNIST images, I used gz archive file. tFor the random number I generated them inside Dataset Class.

* **Combining Two Inputs**

Inside Dataset's getitem(), returned both images and rand_num, along with correct image label and the sum


* **Evaluting Results**
With Validation accuracy of 98%, it seems like model can still be improved. I have randomly added layers in here.

* **Loss Functions**

Used CrossEntropy Loss function, because it's commonly used for classification. They quantify the difference between target and model output distributions.





**Model Keypoints**


![](https://github.com/Gilf641/EVA-6/blob/main/Assignments/S3/model.png)


1. 7 Convolution Blocks
2. BatchNormalization after each layer
3. AvgPool instead of MaxPool before predictions with no ReLU/BN

Validation Accuracy: 98.20% 



**Model Logs**

![](https://github.com/Gilf641/EVA-6/blob/main/Assignments/S3/logs1.png)

![](https://github.com/Gilf641/EVA-6/blob/main/Assignments/S3/logs2.png)

