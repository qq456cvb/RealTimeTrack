//
//  DetectWorker.cpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "DetectWorker.hpp"
#include "TraceManager.hpp"
#include "Timer.hpp"

vector<cv::Point> keypointToPoint(const vector<KeyPoint>& kpts)
{
    vector<cv::Point> result;
    result.resize(kpts.size());
    for (int i = 0; i < kpts.size(); ++i) {
        result[i] = kpts[i].pt;
    }
    return result;
}

cv::Mat concatDescriptors(const std::vector<cv::Mat>& descriptors)
{
    cv::Mat result;
    if (descriptors.size() == 0) {
        return result;
    }
    result = cv::Mat(0, descriptors[0].cols, descriptors[0].type());
    for (int i = 0; i < descriptors.size(); ++i) {
        vconcat(result, descriptors[i], result);
    }
    return result;
}

DetectWorker::DetectWorker(TraceManager* manager) {
    this->delegate = manager;
}


void DetectWorker::detect(cv::Mat &img)
{
    if (!extractor.get()) {
        extractor = ORB::create(inputKeypointSize);
    }
    vector<KeyPoint> rawKeypoints;
    extractor->detect(img, rawKeypoints);
    std::sort(rawKeypoints.begin(), rawKeypoints.end(), [&](cv::KeyPoint p1, cv::KeyPoint p2) {
        return p1.response > p2.response;
    });
    
    if (rawKeypoints.size() < inputKeypointSize) {
        crtKeypoints = rawKeypoints;
    } else {
        crtKeypoints = vector<KeyPoint>(rawKeypoints.begin(), rawKeypoints.begin()+inputKeypointSize);
    }
    extractor->compute(img, crtKeypoints, crtDescriptors);
    {
//        std::unique_lock<std::mutex> lock(delegate->shared_mutex);
//        for (unsigned int i = 0; i < crtKeypoints.size(); i++)
//        {		// Convert to opencv Point object
//            int radius = 2;
//            
//            cv::circle(img, crtKeypoints[i].pt, radius, Scalar::all(200), -1);	// -1 mean filled circle
//        }
    }
}

int DetectWorker::vote(cv::Mat &image)
{
    static Timer timer;
    timer.start();
    cv::flann::Index flannIndex(concatDescriptors(TraceManager::imageDatabase->getAllDescriptors()), cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
    
    
    assert(extractor.get());
    int result = -1;
    const ImageDatabase* database = delegate->imageDatabase;
    const vector<vector<KeyPoint>>& databaseKeypoints = database->getAllKeypoints();
    vector<int> votes;
    
    const std::vector<cv::KeyPoint>& sceneKeypoints = getCrtKeypoints();
    const cv::Mat& sceneDescriptors = getCrtDescriptors();
    cv::Mat results;
    cv::Mat dists;
    
    int dataSize = database->getSize();
    votes.resize(dataSize);
    for (int i = 0; i < votes.size(); i++) {
        votes[i] = 0;
    }
    flannIndex.knnSearch(sceneDescriptors, results, dists, 2, cv::flann::SearchParams());
    
    vector<vector<cv::KeyPoint>> inputGroupKeypoints(dataSize);
    auto acc = TraceManager::imageDatabase->getAccumulates();
    // NDDR method
    float nndrRatio = 0.8f;
    
    
    for (int i = 0; i < sceneDescriptors.rows; ++i)
    {
        if (delegate->inConvex(sceneKeypoints[i].pt)) {
            std::unique_lock<std::mutex> lock(delegate->shared_mutex);
            cv::circle(delegate->outputImg, sceneKeypoints[i].pt, 2, Scalar(255, 0, 0), -1);	// -1 mean filled circle
            continue;
        }
        
        
        //std::cout << dists.at<int>(i, 0) << std::endl;
        if(results.at<int>(i,0) >= 0 && results.at<int>(i,1) >= 0 &&
           dists.at<int>(i,0) <= nndrRatio * dists.at<int>(i,1)) {
            
            int indice = results.at<int>(i,0);

            for (int j = 0; j < dataSize; ++j)
            {
                if (indice >= acc[j] && indice < acc[j+1])
                {
                    votes[j]++;
                    inputGroupKeypoints[j].push_back(sceneKeypoints[i]);
                    break;
                }
            }
        }
    }
    for (int i = 0; i < votes.size(); ++i)
    {
//        std::cout << "votes" << i << " is " << votes[i] << std::endl;
    }
    auto max_vote = std::max_element(votes.begin(), votes.end());
    int distance = (int)std::distance(votes.begin(), max_vote);
    if (*max_vote > THRESHOLD) {
        result = distance;
//        convex = {cv::Point(0, 0), cv::Point(0, 480), cv::Point(640, 480), cv::Point(640, 0)};
//        convexHull(keypointToPoint(inputGroupKeypoints[distance]), convex);
        cout << "Detector select " << distance << "th candidate..." << endl;
    }
    
    timer.stop();
//    cout << "Detect cost " << timer.getElapsedTimeInMilliSec() << " ms\n";
    return result;
}



