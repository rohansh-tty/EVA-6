1. What are Channels and Kernels (according to EVA)?

![Channels](https://www.kdnuggets.com/wp-content/uploads/image-data-analysis.jpg)


Channel is a container of similar features or information. It's a bag of features having same characteristics. Image may have multiple channels.For example, a normal image will have 3 channels like Red, Green, Blue.
Kernels are feature extractors. There are different types of kernels which have specific usecases, like some type of kernels help in detecting vertical edges and some help in detecting horizontal ones.

2. Why should we (nearly) always use 3x3 kernels?

![](https://images1.programmersought.com/213/00/00a8576d530c7c045a288c3dbd00f065.png)

Usually EVEN shaped kernels lack line of symmetry, for example if a 2x2 kernel is used to detect a vertical edge, it can detect edge but it won't have its other portion or there's no symmetry. And this is why most of kernels used are of ODD sized ones i.e 3x3, 5x5 etc. Also 3x3 kernels can act as a base component for large sized kernels. 


|SerialNo |Kernel Size   |No of Params   |  
|---|---|---|
|1   |3x3   |9   |  
|2   |5x5   |25   |
|3   |7x7   |49   |      



3. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199(type each layer output like 199x199 > 197x197...)?

>> 99 times
#PYTHON SCRIPT
convsize = 199
count = 0
while (convsize-2) > 0:
	i += 1
	print(str(convsize)+'x'+str(convsize)+'>'+str(convsize-2)+'x'+str(convsize-2))
	convsize = (convsize-2)


4. How are kernels initialized?
Usually Kernels are randomly initialized. It's not set to zeros, which otherwise would give all input neurons the same weight resulting in same output. Instead Kernels are set to arbitrary values. And later using SGD technique, they are set to optimal values. 

5. What happens during the training of a DNN?

Training of DNN

![](https://databricks.com/wp-content/uploads/2019/02/neural1.jpg)

Based on the problem, we use a certain type of input dataset. This image dataset will be divided in to different classes according to the output. DNN used for Image related problems are called as Convolutional Neural Network. CNN includes Convolutional Layers which convolve the image with specific filters/kernels. These kernels are randomly initialized at first and then are set to different values by Backpropagation process which takes input from the DNN result. CNN also includes Padding and Pooling layers. Padding meanwhile helps in retaining the image size while Pooling reduces image size. 
	


