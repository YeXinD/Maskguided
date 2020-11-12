# Mask-Guided Noise Restriction Adversarial Attacks for Image Classication
This is tensorflow implementation for paper "Mask-Guided Noise Restriction Adversarial Attacks for Image Classication"

First we should get the image saliency map of an input image, codes see "https://github.com/Joker316701882/Salient-Object-Detection", then convert the image saliency map produced by the salient object detection technique to a binary mask, and use the binary mask to restrict the adversarial noise to the salient objects/regions at each iteration. 
We combine the proposed rotation input strategy with iterative attack method to generate stronger adversarial images.


## Pipline
![image](https://github.com/YeXinD/Maskguided/blob/master/pipline.png)

## Sample

The noises of the generated adversarial examples far less visible than the vanilla global noise adversarial examples.

![image](https://github.com/YeXinD/Maskguided/blob/master/sample/sample%201.png)
