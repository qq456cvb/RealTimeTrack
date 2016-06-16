//
//  TraceManager.cpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "TraceManager.hpp"
#include "Configuration.hpp"

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
    this->refMesh = refMesh;
    this->realCamera = realCamera;
    
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
    
    if (Configuration::mode == Configuration::MODE::Extend) {
        needNewWorker = false;
        vector<std::thread> threads;
        threads.resize(workers.size());
        
        // first detect keypoints and descriptors
        // TODO: make detect behind the worker to make convex test right
        detecter->detect(img);
        
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
    }
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

