//
//  TraceManager.hpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef TraceManager_hpp
#define TraceManager_hpp

#include <stdio.h>
#include "LaplacianMesh.h"
#include "Reconstruction.h"
#include "Camera.h"
#include "DefinedMacros.h"
#include "Timer.hpp"
#include "Visualization.h"
#include "BriskKeypointMatcher3D2D.hpp"
#include "TraceWorker.hpp"
#include "DetectWorker.hpp"
#include "ImageDatabase.hpp"
#include "IndepWorker.hpp"
#include <opencv2/opencv.hpp>
#include <armadillo/armadillo>
#include <future>
#include <mutex>

class ConvexHull {
    vector<cv::Point2d> pts;
public:
    bool isInConvexHull(const cv::Point2d& pt);
    int getSize();
    void addPoint(const cv::Point2d& pt);
    void clear();
    void draw(cv::Mat& img);
};

class TraceManager {
    

    const LaplacianMesh* refMesh;
    const Camera* realCamera;
    
    vector<TraceWorker*> workers;
    vector<IndepWorker*> indepWorkers;
    DetectWorker* detecter;
    
    vector<std::mutex*> hullMutex;
    bool needDetect;
    
public:
    static ImageDatabase* imageDatabase;
    vector<ConvexHull> hulls;
    cv::Mat outputImg;
    
    ~TraceManager();
    
    bool needNewWorker;
    void init(LaplacianMesh* refMesh, Camera* realCamera);
    void startNewWorker(int candidateIdx);
    void feed(cv::Mat& img);
    void update();
    int inConvex(const cv::Point2d& pt);
    
    const Camera* getRealCamera();
    const LaplacianMesh* getRefMesh();
    std::mutex shared_mutex;
    
    ConvexHull& getHull(int idx) {
        std::unique_lock<std::mutex> lock(*hullMutex[idx]);
        return  this->hulls[idx];
    }
    
    void setHull(int idx, ConvexHull hull) {
        std::unique_lock<std::mutex> lock(*hullMutex[idx]);
        this->hulls[idx] = hull;
    }
    
    const LaplacianMesh* getRefMesh() const {
        return refMesh;
    }
};

#endif /* TraceManager_hpp */
