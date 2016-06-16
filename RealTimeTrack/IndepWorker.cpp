//
//  IndepWorker.cpp
//  RealTimeTrack
//
//  Created by Neil on 6/2/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "IndepWorker.hpp"
#include "TraceManager.hpp"
#include "Configuration.hpp"

IndepWorker::IndepWorker(TraceManager* manager, int id)
{
    this->delegate = manager;
    this->id = id;
    reconstruction      = new Reconstruction  ( *delegate->getRefMesh(), *delegate->getRealCamera(), wrInit, radiusInit, nUnConstrIters );
    reconstruction->SetUseTemporal(true);
    reconstruction->SetUsePrevFrameToInit(false);
    
    resMesh = *delegate->getRefMesh();
    this->tracker = new InlierMatchTracker();
    crtFrame = 0;
    
    this->refKeypoints = &TraceManager::imageDatabase->getKeypoints(id);
    this->refDescriptors = &TraceManager::imageDatabase->getDescriptors(id);
    this->bary3DRefKeypoints = &TraceManager::imageDatabase->getBary3DRefKeypoints(id);
    
    matchesInlier.clear();
    inlierMatchIdxs.clear();
}

IndepWorker::~IndepWorker()
{
    
}

arma::mat IndepWorker::getMatches(const vector<cv::KeyPoint>& crtKeypoints, const cv::Mat& crtDescriptors)
{
    if (crtKeypoints.size() <= 0) {
        return arma::mat(0, 9);
    }
    static cv::Mat results, dists;
    cv::flann::Index flannIndex(*this->refDescriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
    flannIndex.knnSearch(crtDescriptors, results, dists, 2, cv::flann::SearchParams());
    vector<cv::KeyPoint> inputKeypoints, referenceKeypoints;
    inputKeypoints.reserve(crtKeypoints.size());
    referenceKeypoints.reserve(crtKeypoints.size());
    
    float nndrRatio = 0.8f;
    arma::uvec indices;
    int cnt = 0;
    for (int i = 0; i < crtDescriptors.rows; ++i)
    {
        //std::cout << dists.at<int>(i, 0) << std::endl;
        if(results.at<int>(i,0) >= 0 && results.at<int>(i,1) >= 0 &&
           dists.at<int>(i,0) <= nndrRatio * dists.at<int>(i,1)) {
            int index = results.at<int>(i,0);
            if (index >= (*refKeypoints).size()) {
                continue;
            }
            indices.insert_rows(cnt, 1);
            indices(cnt++) = index;
            inputKeypoints.push_back(crtKeypoints[i]);
            referenceKeypoints.push_back((*refKeypoints)[index]);
        }
    }
    
    arma::mat ref3DPoints = (*bary3DRefKeypoints).rows(indices);
    arma::mat matches3D2D;
    matches3D2D.resize(referenceKeypoints.size(), 9);
    
    arma::mat allPoints = delegate->imageDatabase->getReferenceMesh().GetVertexCoords();
    for(int i = 0; i < referenceKeypoints.size(); i++)
    {
        matches3D2D(i, arma::span(0,5)) = ref3DPoints.row(i);
        matches3D2D(i, 6) = inputKeypoints[i].pt.x;
        matches3D2D(i, 7) = inputKeypoints[i].pt.y;
        matches3D2D(i, 8) = i;
    }
    if (referenceKeypoints.size() <= 0) {
        return arma::mat(0, matches3D2D.n_cols);
    }
    return matches3D2D.rows(0, referenceKeypoints.size()-1);
}

void IndepWorker::trace(cv::Mat &img)
{
    mat trackedMatches, notTrackedMatches, matchesAll;
    if (lostTrack || crtFrame % detectInterval == 0) {
        if (!extractor.get()) {
            extractor = ORB::create();
        }
        vector<KeyPoint> rawKeypoints;
        extractor->detect(img, rawKeypoints);
        std::sort(rawKeypoints.begin(), rawKeypoints.end(), [&](cv::KeyPoint p1, cv::KeyPoint p2) {
            return p1.response > p2.response;
        });
        
        inputKeypoints.clear();
        inputKeypoints.reserve(rawKeypoints.size());
        
        
//        if (rawKeypoints.size() < inputKeypointSize) {
//            inputKeypoints = rawKeypoints;
//        } else {
//            inputKeypoints = vector<KeyPoint>(rawKeypoints.begin(), rawKeypoints.begin()+inputKeypointSize);
//        }
        
        if (lostTrack) {
            for (int i = 0; i < rawKeypoints.size(); i++) {
                if (delegate->inConvex(rawKeypoints[i].pt)) {
                    if (inputKeypoints.size() < inputKeypointSize) {
                        inputKeypoints.push_back(rawKeypoints[i]);
                    }
                }
            }
            extractor->compute(img, inputKeypoints, inputDescriptors);
            notTrackedMatches = getMatches(inputKeypoints, inputDescriptors);
            
            reconstruction->ReconstructPlanarUnconstrIter2D( notTrackedMatches, resMesh, inlierMatchIdxs );
            
            if (inlierMatchIdxs.n_rows > THRESHOLD) {
                lostTrack = false;
                initMatches = (int)inlierMatchIdxs.n_rows;
            }
        } else { // retrieve lost features
            
            for (int i = 0; i < rawKeypoints.size(); ++i) {
                const Point2d& pt = rawKeypoints[i].pt;
                if (delegate->getHull(id).isInConvexHull(pt) && inputKeypoints.size() < inputKeypointSize) {
                    inputKeypoints.push_back(rawKeypoints[i]);
                }
            }
            extractor->compute(img, inputKeypoints, inputDescriptors);
            notTrackedMatches = getMatches(inputKeypoints, inputDescriptors);
        }
        
    }
    if (!lostTrack) {
        {
            std::unique_lock<std::mutex> lock(delegate->shared_mutex);
            trackedMatches = tracker->TrackMatches(img, matchesInlier);
        }
        if (crtFrame % detectInterval == 0) {
            matchesAll = join_vert(notTrackedMatches, trackedMatches);
        } else {
            matchesAll = trackedMatches;
        }
        reconstruction->ReconstructPlanarUnconstrIter2D( matchesAll, resMesh, inlierMatchIdxs );
        
        matchesInlier = matchesAll.rows(inlierMatchIdxs);
//        cout << inlierMatchIdxs.n_rows << endl;
        if (inlierMatchIdxs.n_rows < 0.6 * initMatches) {
            crtFrame = -1;
            lostTrack = true;
            static ConvexHull convex;
            delegate->setHull(id, convex);
            cout << "lost " << crtFrame << endl;
        } else {
            {
                std::unique_lock<std::mutex> lock(delegate->shared_mutex);
                Visualization::DrawPointsAsDot( delegate->outputImg, matchesAll.cols(6,7), KPT_COLOR );				  // All points
                Visualization::DrawPointsAsDot( delegate->outputImg, matchesInlier.cols(6,7), INLIER_KPT_COLOR );// Inlier points
            }
            reconstruction->ReconstructPlanarUnconstrOnce(matchesInlier, resMesh);
            
            if (Configuration::enableDeformTracking) {
                const arma::mat& ctrlVertices = resMesh.GetCtrlVertices();
                vec cInit = reshape( ctrlVertices, resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..
                reconstruction->ReconstructSoftConstr(cInit, resMesh);
            }
            ConvexHull hull;
            static uvec bound_id = {0, 4, 24, 20};
            const arma::mat& boundary = resMesh.GetCtrlVertices().rows(bound_id);
            arma::mat pt2d = delegate->getRealCamera()->ProjectPoints(boundary);
            
            //        cout << "Worker " << id << endl;
            for (int i = 0; i < pt2d.n_rows; i++) {
                hull.addPoint(Point2d(pt2d.at(i, 0), pt2d.at(i, 1)));
            }
            
            {
                std::unique_lock<std::mutex> lock(delegate->shared_mutex);
                Visualization::DrawProjectedMesh(delegate->outputImg, resMesh, *delegate->getRealCamera(), Scalar(0, 0, 255));
            }
            delegate->setHull(id, hull);
        }
    }
    crtFrame++;
}

