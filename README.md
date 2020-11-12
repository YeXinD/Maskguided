# Mask-Guided Noise Restriction Adversarial Attacks for Image Classication
This is tensorflow implementation for paper "Mask-Guided Noise Restriction Adversarial Attacks for Image Classication"

First we should get the image saliency map of an input image, codes see "https://github.com/Joker316701882/Salient-Object-Detection", then convert the image saliency map produced by the salient object detection technique to a binary mask, and use the binary mask to restrict the adversarial noise to the salient objects/regions at each iteration.

##Pipline
![image](https://github.com/YeXinD/Maskguided/blob/master/pipline.png)

##Sample
![image](https://github.com/YeXinD/Maskguided/blob/master/sample/sample%201.png)
