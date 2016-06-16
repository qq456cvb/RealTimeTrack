//
//  IndepWorker.hpp
//  RealTimeTrack
//
//  Created by Neil on 6/2/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef IndepWorker_hpp
#define IndepWorker_hpp

#include <stdio.h>
#include "Reconstruction.h"
#include "InlierMatchTracker.h"
#include <opencv2/opencv.hpp>
#include <armadillo/armadillo>
#include <thread>
using namespace std;
using namespace cv;

class TraceManager;

class IndepWorker {
    Reconstruction* reconstruction;
    TraceManager* delegate;
    InlierMatchTracker* tracker;
    LaplacianMesh resMesh;
    Ptr<ORB> extractor;
    
    const int inputKeypointSize = 1000;
    const arma::mat* bary3DRefKeypoints;
    const vector<cv::KeyPoint>* refKeypoints;
    const cv::Mat* refDescriptors;
    
    vector<cv::KeyPoint> inputKeypoints;
    cv::Mat inputDescriptors;
    int crtFrame;
    arma::uvec					inlierMatchIdxs;
    arma::mat   				matchesInlier;
    unsigned int id;
    
    int initMatches = 0;
    bool lostTrack = true;
    
    /**
     *  when retrieving features, pause the constrcution for one frame, not used now
     */
    bool pausedForRetrieval;
    
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
    
    IndepWorker(TraceManager* manager, int id);
    ~IndepWorker();

    void trace(cv::Mat& img);
};

#endif
