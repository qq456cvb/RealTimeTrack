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
    DetectWorker* detecter;
    
    vector<std::mutex*> hullMutex;
    
    
public:
    static ImageDatabase* imageDatabase;
    vector<ConvexHull> hulls;
    cv::Mat outputImg;
    
    ~TraceManager();
    
    bool needNewWorker;
    void init(LaplacianMesh* refMesh, Camera* realCamera);
    void startNewWorker(int candidateIdx, const vector<cv::Point>& convex);
    void feed(cv::Mat& img);
    void update();
    bool inConvex(const cv::Point2d& pt);
    
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
    
};

#endif /* TraceManager_hpp */
