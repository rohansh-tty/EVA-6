
## Background & Basics



* **Channels**

Image may have single or multiple channels. Channel is similar to a container of **like** features or information. 

For eg: a normal image will have 3 channels like Red, Green, Blue.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRL5E2fITF8Qr1Bz8dPqxC7zr_2QckxzVqHuh8oYTA12hzmU3Nv&usqp=CAU)


* **Kernels**

Kernels are feature extractors. There are different types of kernels which have specific uses, like some type of kernels help in detecting vertical edges, while some help in detecting horizontal ones etc.
For eg: The Kernel below is a Horizontal Edge Detector
![](https://miro.medium.com/max/3146/1*EDqq5ZHYyJE70Zvdt1K_vA.png)

* **Importance of 3x3 Kernel**

Usually the Even sized kernels lack line of symmetry, for example if a 2x2 kernel is used to detect a vertical edge, it can detect edge but it won't have its other portion or there's no symmetry. And this is why most of kernels used are of odd sized ones i.e 3x3, 5x5 etc. Also 3x3 kernels can act as a base component for large sized kernels.

Kernels are randomly initialized. It's not set to zeros, which otherwise would give all input neurons the same weight resulting in same output. Instead Kernels are set to arbitrary values. And later using SGD technique, they are set to optimal values.

* **What is a CNN?**

CNN stands for Convolutional Neural Network, belongs to the class of Deep Neural Networks which is used to analyze visual imagery.

![](https://qph.fs.quoracdn.net/main-qimg-cb67424008b8291ec3fe72dd55ff7171)


![](https://cdn-images-1.medium.com/fit/t/1600/480/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

* **Hierarchy of Feature Detection by CNN**
1. Edges & Gradients
2. Textures and Patterns
3. Parts of Object
4. Object Identifiers

![](https://i.stack.imgur.com/5yGWY.png)

*The left block shows learned kernels in the initial layers. While middle blocks seems to be kernels present in middle. And the right one displays parts of objects learned at final third of layers of convnet. 
This is very important feature of convolutional neural networks.*


* **Receptive Field**: 

The receptive field is defined as the region in the input space that a particular CNN’s feature is looking at. 

![Receptive Field in CNN](https://miro.medium.com/max/4146/1*mModSYik9cD9XJNemdTraw.png)

