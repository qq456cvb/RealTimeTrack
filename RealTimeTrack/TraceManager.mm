//
//  TraceManager.cpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//
#include <Foundation/NSString.h>
#include <Foundation/NSBundle.h>
#include "TraceManager.hpp"
#include "Configuration.hpp"

using namespace cv;

int ConvexHull::getSize()
{
    return (int)pts.size();
}

void ConvexHull::addPoint(const cv::Point2d &pt)
{
    pts.push_back(pt);
}

bool ConvexHull::isInConvexHull(const cv::Point2d& pt)
{
    if (getSize() < 3) {
        return false;
    }
    bool prev_side = false;
    for (int i = 0; i < getSize(); ++i) {
        Point2d& pt1 = pts[i];
        Point2d& pt2 = pts[i == getSize()-1 ? 0 : i+1];
        Point2d line = pt2-pt1;
        Point2d seg = pt-pt1;
        bool side = (line.x*seg.y-line.y*seg.x > 0);
        if (i == 0) {
            prev_side = side;
        } else {
            if (prev_side != side) {
                return false;
            }
        }
    }
    return true;
}

void ConvexHull::clear()
{
    pts.clear();
}

void ConvexHull::draw(cv::Mat &img)
{
    for (int i = 0; i < getSize()-1; ++i) {
        Point2d& pt1 = pts[i];
        Point2d& pt2 = pts[i+1];
        cv::line(img, pt1, pt2, Scalar(0, 255, 0));
    }
    cv::line(img, pts[getSize()-1], pts[0], Scalar(0, 255, 0));
}

ImageDatabase* TraceManager::imageDatabase;

TraceManager::~TraceManager()
{
    for (int i = 0; i < hullMutex.size(); ++i) {
        delete hullMutex[i];
    }
    for (int i = 0; i < hulls.size(); ++i) {
        hulls[i].clear();
    }
    delete detecter;
    for (int i = 0; i < workers.size(); ++i) {
        delete workers[i];
    }
}

void TraceManager::init(LaplacianMesh *refMesh, Camera *realCamera)
{
    NSString *vocPath, *paramPath;
    vocPath = [[NSBundle mainBundle] pathForResource:@"ORBvoc" ofType:@"txt"];
    paramPath = [[NSBundle mainBundle] pathForResource:@"param" ofType:@"yaml"];
    this->refMesh = refMesh;
    this->realCamera = realCamera;
//    this->slam = std::make_shared<ORB_SLAM2::System>(std::string([vocPath UTF8String]),std::string([paramPath UTF8String]),ORB_SLAM2::System::MONOCULAR, false);
    
    if (Configuration::mode == Configuration::MODE::Extend)
    {
        detecter = new DetectWorker(this);
        needDetect = true;
    }
}

void TraceManager::feed(cv::Mat &img)
{
    if (this->imageDatabase->getSize() == 0) {
        return;
    }
    static Timer t,totalT;
    totalT.start();
    if (Configuration::mode == Configuration::MODE::Extend) {
        needNewWorker = false;
        vector<std::thread> threads;
        threads.resize(workers.size());
        
        // first detect keypoints and descriptors
        // TODO: make detect behind the worker to make convex test right
        
        
        t.start();
        detecter->detect(img);
        t.stop();
        cout << "Detect costs " << t.getElapsedTimeInMilliSec() << " ms\n";
        
        std::future<int> future;
        //    detecter->vote(img);
        if (needDetect) {
            future = std::async(std::launch::async, &DetectWorker::vote, detecter, std::ref(img));
        }
        //    int candidateIdx = detecter->vote(img);
        
        // reuse detector's descriptor
        for (int i = 0; i < threads.size(); i++) {
            if (workers[i]->status == TraceWorker::BUSY) {
                threads[i] = thread(&TraceWorker::trace, workers[i], std::ref(img), detecter);
            }
        }
        
        // wait for all tasks to finish
        for (int i = 0; i < threads.size(); i++) {
            if (threads[i].joinable()) {
                threads[i].join();
            }
        }
        if (needDetect) {
            int candidateIdx = future.get();
            needNewWorker = candidateIdx != -1;
            
            
            if (needNewWorker) {
                startNewWorker(candidateIdx);
                needDetect = false;
            }
        } else {
            needDetect = true;
        }
    } else {
//        std::thread slamThread = thread(&TraceManager::slamRun, this, std::ref(img));
        if (indepWorkers.size() == 0) {
            for (int i = 0; i < this->imageDatabase->getSize(); i++) {
                indepWorkers.push_back(new IndepWorker(this, i));
                hullMutex.push_back(new std::mutex);
                hulls.push_back(ConvexHull());
            }
        }
        vector<std::thread> threads;
        threads.resize(indepWorkers.size());
        
        for (int i = 0; i < threads.size(); i++) {
            threads[i] = thread(&IndepWorker::trace, indepWorkers[i], std::ref(img));
        }
        for (int i = 0; i < threads.size(); i++) {
            if (threads[i].joinable()) {
                threads[i].join();
            }
        }
        bool willLost = false;
        for (int i = 0; i < threads.size(); i++) {
            if (indepWorkers[i]->willLost) {
                willLost = true;
                break;
            }
        }
//        if (slamThread.joinable()) {
//            slamThread.join();
//        }
//        currentPose = slam->GetCurrentPose();
//        if (!currentPose.empty()) {
//            if (willLost) {
//                if (refPose.empty()) {
//                    refPose = currentPose.clone();
//                }
//                cv::Mat R1 = currentPose(cv::Range(0, 3), cv::Range(0, 3));
//                cv::Mat T1 = currentPose(cv::Range(0, 3), cv::Range(3, 4));
//                
//                cv::Mat R2 = refPose(cv::Range(0, 3), cv::Range(0, 3));
//                cv::Mat T2 = refPose(cv::Range(0, 3), cv::Range(3, 4));
//                cv::Mat R = R1 * R2.t();
//                
//                // multiply by scale due to the descriptor scale
//                cv::Mat T = (T1 - T2) * 100;
//                
//                cv::hconcat(R, T, relativePose);
//            } else {
//                refPose = cv::Mat();
//                relativePose = cv::Mat();
//            }
//        }
        
    }
    totalT.stop();
//    cout << "total time " << totalT.getElapsedTimeInMilliSec() << " ms\n";
}

void TraceManager::slamRun(const cv::Mat &img)
{
    slam->TrackMonocular(img, time(nullptr));
}

void TraceManager::startNewWorker(int candidateIdx)
{
    cout << "Starting worker...\n";
    // find available worker
    bool reuseWorker = false;
    for (int i = 0; i < workers.size(); i++) {
        if (workers[i]->status == TraceWorker::AVAILABLE) {
            workers[i]->reset(refMesh, this->imageDatabase->getBary3DRefKeypoints(candidateIdx), this->imageDatabase->getKeypoints(candidateIdx), this->imageDatabase->getDescriptors(candidateIdx));
            reuseWorker = true;

            break;
        }
    }
    
    // no available worker, create new worker
    if (!reuseWorker) {
        TraceWorker* worker = new TraceWorker(this, (int)workers.size());
        workers.push_back(worker);
        worker->reset(refMesh, this->imageDatabase->getBary3DRefKeypoints(candidateIdx), this->imageDatabase->getKeypoints(candidateIdx), this->imageDatabase->getDescriptors(candidateIdx));
        hullMutex.push_back(new std::mutex);
        hulls.push_back(ConvexHull());
        
    }
}

int TraceManager::inConvex(const cv::Point2d &pt)
{
    for (int i = 0; i < hulls.size(); ++i) {
        if (this->getHull(i).isInConvexHull(pt)) {
            return i;
        }
    }
    return -1;
}

const Camera* TraceManager::getRealCamera()
{
    return realCamera;
}

const LaplacianMesh* TraceManager::getRefMesh()
{
    return refMesh;
}

