# RealTimeTrack

<!-- README refined by Cursor -->

A fast algorithm tracking real time deformable planers

## Overview

This repository contains C++, Objective-C++, MATLAB/Objective-C, C/C++ code from an older research, course, or prototype project. The README has been refreshed to make the repository easier to scan while preserving the original notes below.

## Repository Contents

- `RealTimeTrack/`
- `RealTimeTrack.xcodeproj/`
- `RealTimeTrackTests/`
- `RealTimeTrackUITests/`

## Setup

- Open the `.xcodeproj` project in Xcode for iOS/macOS builds.

## Usage

- inspect the source directories listed below; many of these older repos were kept as research prototypes rather than packaged applications.

## Data and Artifacts

No new large artifact is stored in this repository. If a dataset or checkpoint is required, follow the links and notes in the original section below.

## Status

This is a `Batch B` cleanup pass for a legacy repository. Commands may require dependency/version adjustments on a modern machine.

## License

No explicit license file was found in this checkout; check the original project context before reusing code.

## Original Notes

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

# Configurations
- enableDeformTracking will build up a soft plane but hurt performance
- MODE has two values
  - Default can track different objects and is faster
  - Extend can track identical objects but loses performance

# Unity Plugin
I also export a unity plugin package for this code so that it can be used in Unity.
You can download it from [here](http://youyangsoft.com/AR.unitypackage)
