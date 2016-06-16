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

cv::Mat* concatDescriptors(const std::vector<cv::Mat>& descriptors)
{
    cv::Mat* result = NULL;
    if (descriptors.size() == 0) {
        return result;
    }
    result = new cv::Mat(0, descriptors[0].cols, descriptors[0].type());
    for (int i = 0; i < descriptors.size(); ++i) {
        vconcat(*result, descriptors[i], *result);
    }
    return result;
}

DetectWorker::DetectWorker(TraceManager* manager) {
    this->delegate = manager;
    databaseDesc = nullptr;
}

DetectWorker::~DetectWorker()
{
    if (this->extractor.get()) {
        extractor.release();
    }
    if (this->databaseDesc) {
        delete databaseDesc;
        databaseDesc = nullptr;
    }
}

void DetectWorker::detect(cv::Mat &img)
{
    if (!extractor.get()) {
        extractor = ORB::create();
    }
    vector<KeyPoint> rawKeypoints;
    extractor->detect(img, rawKeypoints);
    std::sort(rawKeypoints.begin(), rawKeypoints.end(), [&](cv::KeyPoint p1, cv::KeyPoint p2) {
        return p1.response > p2.response;
    });
    
    nonfreeKeypoints.clear();
    freeKeypoints.clear();
    nonfreeKeypoints.resize(delegate->hulls.size());
    freeKeypoints.reserve(rawKeypoints.size());
    for (int i = 0; i < rawKeypoints.size(); i++) {
        int id = delegate->inConvex(rawKeypoints[i].pt);
        if (id != -1) {
            if (nonfreeKeypoints[id].size() < inputKeypointSize) {
                nonfreeKeypoints[id].push_back(rawKeypoints[i]);
            }
            
        } else {
            if (freeKeypoints.size() < inputKeypointSize) {
                freeKeypoints.push_back(rawKeypoints[i]);
            }
        }
    }
//    if (freeKeypoints.size() > inputKeypointSize) {
//        freeKeypoints.erase(freeKeypoints.begin() + inputKeypointSize, freeKeypoints.end());
//    }
//    if (rawKeypoints.size() < inputKeypointSize) {
//        crtKeypoints = rawKeypoints;
//    } else {
//        crtKeypoints = vector<KeyPoint>(rawKeypoints.begin(), rawKeypoints.begin()+inputKeypointSize);
//    }
//    extractor->compute(img, crtKeypoints, crtDescriptors);
//    while (!crtDescriptors.isContinuous()) {
//        crtDescriptors = crtDescriptors.clone();
//    }
//    assert(crtDescriptors.isContinuous());
//    {
//        std::unique_lock<std::mutex> lock(delegate->shared_mutex);
//        for (unsigned int i = 0; i < crtKeypoints.size(); i++)
//        {		// Convert to opencv Point object
//            int radius = 2;
//            
//            cv::circle(img, crtKeypoints[i].pt, radius, Scalar::all(200), -1);	// -1 mean filled circle
//        }
//    }
}

int DetectWorker::vote(cv::Mat &image)
{
    Timer timer;
    timer.start();
//    std::shared_ptr<cv::flann::Index> flannIndex = std::make_shared<cv::flann::Index>(concatDescriptors(TraceManager::imageDatabase->getAllDescriptors()), cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
//    auto desc = getCrtDescriptors().clone();
    if (databaseDesc == nullptr) {
        databaseDesc = concatDescriptors(TraceManager::imageDatabase->getAllDescriptors());
    }
    cv::flann::Index flannIndex(*databaseDesc, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
//    const std::vector<cv::Mat>& bigMat = TraceManager::imageDatabase->getAllDescriptors();
//    cv::Mat test;
//    test = cv::Mat(0, bigMat[0].cols, bigMat[0].type());
//    for (int i = 0; i < bigMat.size(); ++i) {
//        vconcat(test, bigMat[i], test);
//    }
//    FlannBasedMatcher matcher;
//    std::vector< DMatch > matches;
//    matcher.match( bigMat, getCrtDescriptors(), matches );
    assert(extractor.get());
    int result = -1;
    const ImageDatabase* database = delegate->imageDatabase;
//    const vector<vector<KeyPoint>>& databaseKeypoints = database->getAllKeypoints();
    vector<int> votes;
    
    extractor->compute(image, freeKeypoints, crtFreeDescriptors);
    cv::Mat results;
    cv::Mat dists;
    
    int dataSize = database->getSize();
    votes.resize(dataSize);
    for (int i = 0; i < votes.size(); i++) {
        votes[i] = 0;
    }
    flannIndex.knnSearch(crtFreeDescriptors, results, dists, 2, cv::flann::SearchParams());
    
    vector<vector<cv::KeyPoint>> inputGroupKeypoints(dataSize);
    auto acc = TraceManager::imageDatabase->getAccumulates();
    // NDDR method
    float nndrRatio = 0.8f;
    
    
    for (int i = 0; i < crtFreeDescriptors.rows; ++i)
    {
        if (delegate->inConvex(freeKeypoints[i].pt) != -1) {
//            std::unique_lock<std::mutex> lock(delegate->shared_mutex);
//            cv::circle(delegate->outputImg, sceneKeypoints[i].pt, 2, Scalar(255, 0, 0), -1);	// -1 mean filled circle
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
                    inputGroupKeypoints[j].push_back(freeKeypoints[i]);
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
//    static int cnt = 0;
//    static double averageTime = 0;
//    averageTime = (cnt*averageTime + timer.getElapsedTimeInMilliSec()) / (cnt+1);
//    cnt++;
//    cout << "average time " << cnt << " :" << averageTime << " ms\n";
    return result;
}



