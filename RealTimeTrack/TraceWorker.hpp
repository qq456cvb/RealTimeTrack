//
//  TraceWorker.hpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef TraceWorker_hpp
#define TraceWorker_hpp

#include <stdio.h>
#include "Reconstruction.h"
#include "InlierMatchTracker.h"
#include <opencv2/opencv.hpp>
#include <armadillo/armadillo>
#include <thread>
using namespace std;

class TraceManager;

class TraceWorker {
    Reconstruction* reconstruction;
    TraceManager* delegate;
    InlierMatchTracker* tracker;
    LaplacianMesh resMesh;
    const arma::mat* bary3DRefKeypoints;
    const vector<cv::KeyPoint>* refKeypoints;
    const cv::Mat* refDescriptors;
    int crtFrame;
    arma::uvec					inlierMatchIdxs;
    arma::mat   				matchesInlier;
    unsigned int id;
    
    int initMatches = 0;
    
    /**
     *  recompute keypoints' interval, per 10 frames
     */
    const int     nUnConstrIters  = 5;
    const int     radiusInit        = 5  * pow(Reconstruction::ROBUST_SCALE, nUnConstrIters-1);
    const double  wrInit          = 525 * pow(Reconstruction::ROBUST_SCALE, nUnConstrIters-1);
    
    const int detectInterval = 10;
    
    const int THRESHOLD = 100;
    
    arma::mat getMatches(const vector<cv::KeyPoint>& crtKeypoints, const cv::Mat& crtDescriptors);
    
public:
    enum STATUS {
        INIT,
        AVAILABLE,
        BUSY,
        DEAD
    };
    
    TraceWorker(TraceManager* manager, int id);
    ~TraceWorker();
    
    STATUS status;
    void stop();
    void reset(const LaplacianMesh* refMesh, const arma::mat& refKeypoints3d, const vector<cv::KeyPoint>& refKeypoints2d, const cv::Mat& descriptors);
    void trace(cv::Mat& img, const vector<cv::KeyPoint>& crtKeypoints, const cv::Mat& crtDescriptors);
};

#endif /* TraceWorker_hpp */
