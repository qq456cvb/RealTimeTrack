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
    
    std::vector<KeyPoint> crtKeypoints;
    cv::Mat crtDescriptors;
    const int inputKeypointSize = 1000;
    const int THRESHOLD = 150;
    
public:
    DetectWorker(TraceManager* manager);
    const std::vector<KeyPoint>& getCrtKeypoints() {
        return crtKeypoints;
    };
    const cv::Mat& getCrtDescriptors() {
        return crtDescriptors;
    };
    
    /**
     *  detect, -1 if no candidate, else return candidate index in database
     *
     *  @param img current image
     */
    int vote(cv::Mat& img);
    void detect(cv::Mat& img);
};
#endif /* DetectWorker_hpp */
