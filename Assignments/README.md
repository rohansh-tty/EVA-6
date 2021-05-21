## S3 Assignment

**Main Task**
   Write a neural network that can:
    * take 2 inputs:
        * an image from MNIST dataset, and
        * a random number between 0 and 9
    * and gives two outputs:
        * the "number" that was represented by the MNIST image, and
        * the "sum" of this number with the random number that was generated and sent as the input to the network

###### Assignment Solution: ![S4-Solution](https://github.com/Gilf641/EVA4/blob/master/S4/S4-Assignment-Solution.ipynb)


well documented (via readme file on github and comments in the code)
must mention the data representation
must mention your data generation strategy
must mention how you have combined the two inputs
must mention how you are evaluating your results
must mention "what" results you finally got and how did you evaluate your results
must mention what loss function you picked and why!
training MUST happen on the GPU

Steps:

Dataset Representation
I have used idx3-ubyte format to store and save MNIST Dataset. It's loaded as an archive file(.gz format). 

Dataset Generation strategy
For MNIST images, I used gz archive file. tFor the random number I generated them inside Dataset Class.

Combining Two Inputs
Inside Dataset's getitem(), returned both images and rand_num, along with correct image label and the sum

Evaluting Results


Loss Functions
Used CrossEntropy Loss function, because it's commonly used for classification. They quantify the difference between target and model output distributions.





**Model Keypoints**
1. 3 Convolution Blocks
2. BatchNormalization after each layer
3. Dropout at the end of each block
4. AvgPool instead of MaxPool
5. 17,594 Parameters
5. Global Average Pooling at last

Validation Accuracy: 99.42% 

