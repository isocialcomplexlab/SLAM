<h1 align="center"> 
	Loop Closure Detection System
</h1>


Content
=================
<!--ts-->
   * [Description](#Description)
   * [Files](#Files)
   * [Videos](#Videos)
   * [Results](#Results)
   * [Scholarships Received](#Scholarships-Received)
<!--te-->

---
Description 
=================
Master's degree in Computer Science at the Federal University of ABC;

Title of Work: Loop Closure Detection in Visual SLAM Based on Convolutional Neural Network;

Student: Fabiana Naomi Iegawa;

Advisor: Wagner Tanaka Botelho;

---
Papers
=================
## ITNG 2023

Title: Loop Closure Detection in Visual SLAM Based on Convolutional Neural Network

Authors: Fabiana Naomi Iegawa, Wagner Tanaka Botelho, Tamires dos Santos, Edson Pinheiro Pimentel, Flavio Shigeo Yamamoto

link: https://link.springer.com/book/10.1007/978-3-031-28332-1?sap-outbound-id=D653B47018FE3DA2A47097DFE440FA7C63F99813

---
Files 
=================
## CNN
CNN_vgg16.ipynb -> Code implemented for CNN training;

## Gazebo_ROS
/ros_control/move.py -> Code implemented to control robot's movements;

/ros_control/model_state.py -> Code implemented to capture and save images during navigation;

/dataset -> Images captured during navigation;

## System
LoopClosureGazeboOffice_256.ipynb -> System implemented;

Grafico3D.ipynb -> Generate positions' graph;

/tensors_gazebo -> Tensors Database;

/artifacts/jackal_office_gazebo.mp4 -> Video of the navigation on Gazebo;

/artifacts/office_gazebo_model_state.txt -> Robot's positions during navigation;

/artifacts/loop.txt -> Loop closure detected according to the image;

## KITTI

/Group5 -> Results for Group 5 based on each NN (AlexNet, ConvNeXt, ResNet, Original VGG-16, Adapted and Trained VGG-16, and VisionTransformer)

/Group8 -> Results for Group 8 based on each NN (AlexNet, ConvNeXt, ResNet, Original VGG-16, Adapted and Trained VGG-16, and VisionTransformer)

## TUM

Results based on each NN (AlexNet, ConvNeXt, ResNet, Original VGG-16, Adapted and Trained VGG-16, and VisionTransformer)

---
Video
=================
Link to YouTube video of the robot navigation in Gazebo-> https://youtu.be/_FArjxUQ7l8

---
Results
=================
Navigation path in Gazebo office scenario.

<p align="center">
  <img alt="Scenario" title="#Gazebo Office Scenario" src="./assets/new_scene.png" width="700px">
</p>

---
Acknowledgment
=================
This work was supported by a Technical Training Fellowship (TT-3/Process Number: 2019/12080-5) funded by the São Paulo Research Foundation (FAPESP)/PIPE Grant Program from July/2019 to December/2020 offered by the Startup NTU Software Technology (Process Number: 2018/04306-0).

