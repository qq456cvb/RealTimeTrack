# RealTimeTrack
A fast algorithm tracking real time deformable planers.

This is part of my graduation design. It can track multiple images at the same time at a FPS above 30.

# Demo
[![video](http://img.youtube.com/vi/8OSTpl4fAJ0/0.jpg)](https://www.youtube.com/watch?v=8OSTpl4fAJ0)
# Requirements
- libarmadillo, already included
- opencv 3+ for iOS

# TODO
I try to add ORB-SLAM into it to be able to predict the object when lost, while not finished yet. 
If you'd like to use it and have a try, please include g2o and DBoW2 for iOS. 
You can either compile the src file or build a framework for g2o or DBoW2.
For more info, see [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)

# Paper
[Real-time Track](http://youyangsoft.com/paper.pdf)

# Configurations
- enableDeformTracking will build up a soft plane but hurt performance
- MODE has two values
  - Default can track different objects and is faster
  - Extend can track identical objects but loses performance

# Unity Plugin
I also export a unity plugin package for this code so that it can be used in Unity.
You can download it from [here](http://youyangsoft.com/AR.unitypackage)
