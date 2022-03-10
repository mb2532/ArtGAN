# ArtGAN

<!-- ABOUT THE PROJECT -->


![gui](/images/results.png)

Welcome to my CS 236G Final Project 

In this project, we train CycleGANs to perform photo-to-painting style transfer to create face filters that transform real images into the style of several painting genres. 

In this repository you'll find:
* A PyTorch implementation of the CycleGAN architecture
* Code to run the CycleGAN training process 
* Face image and painting datasets 
* Pretrained PyTorch models for the cubism and pop art ArtGANs
* A script to launch an interface to try out pretrained ArtGANs


## Contents


1. `cubism_cyclegan_training` : code to implement CycleGAN and run training process using cubism dataset 
3. `popart_cyclegan_training` : code to implement CycleGAN and run training process using pop art dataset 
4. `preprocess` : contains script to upsample trainA images 
5. `artgan-interface` : code to launch interface to use pretrained ArtGANs


## Launching the ArtGAN Interface

You'll first need to clone this repository `mb2532/ArtGAN` and navigate to the `artgan-interface` directory. 

To launch the user interface run the script: 
  ```sh
  python facefiltergui.py
  ```
The following interface should appear: 
![gui](/images/artgan_interface.png)

Hit "Browse" and select the directory containing your desired face image (as a .png file). 
Select an art genre from the bottom, and hit generate!


