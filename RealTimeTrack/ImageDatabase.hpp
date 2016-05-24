//
//  ImageDatabase.hpp
//  RealTimeTrack
//
//  Created by Neil on 5/6/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef ImageDatabase_hpp
#define ImageDatabase_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "LaplacianMesh.h"
#include "Camera.h"
#include <armadillo/armadillo>
using namespace cv;
using namespace arma;

class ImageDatabase {
    int size;
    const int refDescriptorSize = 1000;
    // of the reference planar object
    const LaplacianMesh &referenceMesh;
    
    // Camera object in camera coordinate
    const Camera& camCamera;
    
    Ptr<ORB> extractor;
    
    std::vector<mat> bary3DRefKeypoints, bary2DRefKeypoints;
    std::vector<std::vector<KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    std::vector<int> accumulates;
    
public:
    ImageDatabase(const LaplacianMesh &, const Camera &);
    
    int getSize() const { return size; };
    const std::vector<KeyPoint>& getKeypoints(int index) const;
    const cv::Mat& getDescriptors(int index) const;
    const arma::mat& getBary3DRefKeypoints(int index) const;
    const std::vector<cv::Mat>& getAllDescriptors() const;
    const std::vector<std::vector<KeyPoint>>& getAllKeypoints() const;
    const std::vector<int>& getAccumulates() const;
    const Camera& getRefCamera() const;
    const LaplacianMesh& getReferenceMesh() const;
    
    void addImage(const cv::Mat& img);
    bool find3DPointOnMesh(const cv::Point2d& refPoint, rowvec& intersectionPoint) const;
    vec findIntersectionRayTriangle(const vec& source, const vec& destination, const mat& vABC) const;
};
#endif /* ImageDatabase_hpp */
