# RealTimeTrack

An iOS app that detects and tracks multiple **deformable planar surfaces** (paper sheets, posters, cloth) in real time — above 30 FPS on-device. Built as my undergraduate thesis project.

[![video](http://img.youtube.com/vi/8OSTpl4fAJ0/0.jpg)](https://www.youtube.com/watch?v=8OSTpl4fAJ0)

## How It Works

The pipeline combines wide-baseline detection with frame-to-frame tracking, running each tracked target on its own worker thread:

- **Detection** (`DetectWorker`): candidate targets are recognized with BRISK/ORB keypoint matching against registered template images (`KeypointMatcher3D2D` and variants); an `ImageDatabase` built on [DBoW2](https://github.com/dorian3d/DBoW2) provides fast image retrieval.
- **Tracking** (`TraceWorker` / `IndepWorker`, orchestrated by `TraceManager`): matched keypoints are followed with Lucas-Kanade optical flow (`LKPointTracker`) and filtered by `InlierMatchTracker`, so each target keeps tracking without re-detection.
- **Deformable reconstruction**: the planar template is modeled as a Laplacian triangle mesh (`LaplacianMesh`, `TriangleMesh`) whose 3D shape is recovered every frame by constrained linear least-squares (`SoftConstrOptimize`, `EqualConstrOptimize`, `IneqConstrOptimize`) using [Armadillo](http://arma.sourceforge.net/) (an iOS build is bundled).
- **Relocalization (WIP)**: `Src/` contains an [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) port (with bundled g2o and DBoW2) intended to predict the object pose when tracking is lost; the integration was never finished.

## Requirements

- Xcode (the repo is a complete `RealTimeTrack.xcodeproj` iOS project)
- OpenCV 3+ for iOS (`opencv2.framework`)
- Armadillo — already included (`armadillo.framework` / `libarmadillo-ios.a`)

## Usage

Open `RealTimeTrack.xcodeproj` in Xcode, add `opencv2.framework`, and run on a device. Register a template by pointing the camera at the target and capturing it (see `ImageChoose` / `ViewController`), then the app detects and tracks it live.

### Configuration (`Configuration.hpp`)

- `enableDeformTracking` — reconstructs the deforming 3D mesh of the surface (softer, more expensive); disable for rigid planar tracking.
- `mode`:
  - `Default` — tracks multiple *different* objects, fastest.
  - `Extend` — also tracks multiple *identical* objects, at some performance cost.

## Unity Plugin

A Unity plugin export of this tracker is available [here](http://youyangsoft.com/AR.unitypackage).
