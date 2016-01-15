//
//  ImageChoose.hpp
//  RealTimeTrack
//
//  Created by Neil on 1/6/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef ImageChoose_hpp
#define ImageChoose_hpp

#include <armadillo/armadillo>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>
#include "Camera.h"
#include "LaplacianMesh.h"
#include "Timer.hpp"

using namespace cv;
using namespace std;
using namespace arma;
class ImageChoose {
    Timer timer;
    
    vector<int> votes;
    vector<int> sizes;
    
    const int refDescriptorSize = 500, inputDescriptorSize = 500;
    cv::Mat descriptors;
    vector<vector<cv::KeyPoint>> keypoints;
    Ptr<BRISK> detector;
    // Reference triangular mesh in CAMERA coordinate that is 3D coordinate
    // of the reference planar object
    const TriangleMesh &referenceMesh;
    
    // Camera object in camera coordinate
    const Camera& camCamera;
    mat bary3DRefKeypoints;
public:
    ImageChoose(const TriangleMesh&, const Camera&);
    ~ImageChoose();
    void StoreDescriptors(const cv::Mat& image);
    const cv::Mat getDescriptors(int index);
    const vector<cv::KeyPoint> getKeyPoints(int index);
    int Vote(const cv::Mat& image, vector<cv::KeyPoint>& refKeypoints, vector<cv::KeyPoint>& inputKeypoints, mat& ref3DPoints);
    bool find3DPointOnMesh(const cv::Point2d& refPoint, rowvec& intersectionPoint) const;
    vec findIntersectionRayTriangle(const vec& source, const vec& destination, const mat& vABC) const;
};

#endif /* ImageChoose_hpp */
