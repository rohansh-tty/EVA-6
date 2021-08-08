# S11 Assignment



# ASSIGNMENT A

    OpenCV Yolo: SOURCE (Links to an external site.)
        Run this above code on your laptop or Colab. 
        Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
        Run this image through the code above. 
        Upload the link to GitHub implementation of this
        Upload the annotated image by YOLO. 


[Notebook Link](https://github.com/Gilf641/EVA-6/blob/master/Assignments/S11/Assignment-A/S11_Assignment_A.ipynb)


* **IMPLEMENTATION**
1. Ran the OpenCV-Yolo Colab Notebook
2. Uploaded a couple of images with me holding a some COCO Objects and tested the same








# ASSIGNMENT B

Training Custom Dataset on Colab for YoloV3
        
        Refer to this Colab File: LINK (Links to an external site.)
        Refer to this GitHub Repo (Links to an external site.)
        Collect a dataset of 500 images and annotate them. Please select a class for which you can find a YouTube video as well. Steps are explained in the readme.md file on GitHub.
        Once done:
            Download (Links to an external site.) a very small (~10-30sec) video from youtube which shows your class. 
            Use ffmpeg (Links to an external site.) to extract frames from the video. 
            Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
            Inter on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
            `python detect.py --conf-thres 0.3 --output output_folder_name`
            Use ffmpeg (Links to an external site.) to convert the files in your output folder to video
            Upload the video to YouTube. 

            
[Repo Link](https://github.com/Gilf641/EVA-6/tree/master/Assignments/S11/Assignment-B)


[Notebook_Link](https://github.com/Gilf641/EVA-6/blob/master/Assignments/S11/Assignment-B/S11_Assignment_B.ipynb)

---------

**RESULTS**

![](/Assignments/S11/Assignment-B/assets/out_out/out20.png)


# Model Logs

Model Summary: 225 layers, 6.25733e+07 parameters, 6.25733e+07 gradients
Caching labels (510 found, 0 missing, 0 empty, 0 duplicate, for 510 images): 100% 510/510 [00:00<00:00, 9034.83it/s]
Caching images (0.3GB): 100% 510/510 [00:01<00:00, 464.56it/s]
Caching labels (510 found, 0 missing, 0 empty, 0 duplicate, for 510 images): 100% 510/510 [00:00<00:00, 10415.30it/s]
Caching images (0.3GB): 100% 510/510 [00:01<00:00, 457.72it/s]
Image sizes 512 - 512 train, 512 test
Using 2 dataloader workers
Starting training for 160 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     0/159     9.16G      4.01      3.68         0      7.69        11       512: 100% 43/43 [00:57<00:00,  1.34s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:18<00:00,  2.29it/s]
                 all       510       519         0         0     0.345         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     1/159     9.17G      3.11      1.94         0      5.05        16       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:14<00:00,  3.05it/s]
                 all       510       519      0.29     0.886     0.478     0.437

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     2/159     9.17G      2.94      1.22         0      4.16        15       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.27it/s]
                 all       510       519      0.34     0.879     0.415      0.49

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     3/159     9.17G      2.39     0.918         0      3.31        10       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.25it/s]
                 all       510       519     0.284     0.906     0.328     0.433

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     4/159     9.17G      2.29     0.816         0       3.1        12       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.26it/s]
                 all       510       519     0.828     0.952     0.943     0.886

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     5/159     9.17G      2.19     0.759         0      2.95        14       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.27it/s]
                 all       510       519    0.0831     0.956     0.135     0.153

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     6/159     9.17G      1.94     0.772         0      2.71         9       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.25it/s]
                 all       510       519    0.0302     0.884    0.0702    0.0584

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     7/159     9.17G      1.78     0.727         0      2.51        16       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.27it/s]
                 all       510       519     0.851     0.965     0.971     0.905

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     8/159     9.17G      1.96     0.713         0      2.67        11       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.951     0.967     0.977     0.959

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     9/159     9.17G      1.56     0.726         0      2.29        12       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.941     0.934     0.965     0.938

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    10/159     9.17G      1.79     0.714         0       2.5        11       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.912     0.919     0.962     0.916

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    11/159     9.17G      1.84     0.678         0      2.52        25       512:  63% 27/43 [00:35<00:21,  1.32s/it]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.02+/-0.22      -4.99+/-0.90       4.11+/-0.01 
                         101       0.18+/-0.27      -6.49+/-0.31       4.13+/-0.00 
                         113       0.01+/-0.10      -6.04+/-0.36       4.09+/-0.01 
    11/159     9.17G      1.66     0.683         0      2.35        16       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.901     0.971     0.971     0.935

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    12/159     9.17G      1.33     0.675         0      2.01        11       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.27it/s]
                 all       510       519      0.84      0.94      0.95     0.887

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    13/159     9.17G      1.51      0.68         0      2.19        15       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.29it/s]
                 all       510       519     0.964     0.917      0.96      0.94

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    14/159     9.17G      1.33     0.586         0      1.91        10       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.937     0.828     0.933     0.879

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    15/159     9.17G      1.45     0.619         0      2.07        13       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.29it/s]
                 all       510       519     0.937     0.923     0.957      0.93

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    16/159     9.17G      1.42     0.649         0      2.07        17       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519      0.91     0.965     0.976     0.937

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    17/159     9.17G      1.24     0.617         0      1.86        12       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.956     0.969     0.988     0.962

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    18/159     9.17G      1.49     0.621         0      2.11        10       512: 100% 43/43 [00:56<00:00,  1.32s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.29it/s]
                 all       510       519     0.974     0.963     0.991     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    19/159     9.17G      1.16     0.606         0      1.76        11       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.28it/s]
                 all       510       519     0.951     0.967     0.988     0.959

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    20/159     9.17G      1.15     0.627         0      1.78        13       512: 100% 43/43 [00:56<00:00,  1.31s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 43/43 [00:13<00:00,  3.29it/s]
                 all       510       519     0.963     0.985     0.992     0.974


