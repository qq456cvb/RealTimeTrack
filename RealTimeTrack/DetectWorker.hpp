//
//  DetectWorker.hpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef DetectWorker_hpp
#define DetectWorker_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#include <mutex>

class TraceManager;

class DetectWorker {
    TraceManager* delegate;
    Ptr<ORB> extractor;
    
    // those in convex
    std::vector<std::vector<KeyPoint>> nonfreeKeypoints;
    
    // not inconvex
    std::vector<KeyPoint> freeKeypoints;
    cv::Mat* databaseDesc;
    cv::Mat crtFreeDescriptors;
    const int inputKeypointSize = 1000;
    const int THRESHOLD = 100;
    
public:
    DetectWorker(TraceManager* manager);
    ~DetectWorker();
    const std::vector<KeyPoint>& getNonfreeKeypoints(int idx) {
        return nonfreeKeypoints[idx];
    };
    const cv::Mat& getFreeDescriptors() {
        return crtFreeDescriptors;
    };
    
    const std::vector<KeyPoint>& getFreeKeypoints() {
        return freeKeypoints;
    };
    
    /**
     *  vote, -1 if no candidate, else return candidate index in database
     *
     *  @param img current image
     */
    int vote(cv::Mat& img);
    
    /**
     *  assign feature points to its own regions
     *
     *  @param img current image
     */
    void detect(cv::Mat& img);
};
#endif /* DetectWorker_hpp */
