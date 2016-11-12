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
#include <dispatch/dispatch.h>

IndepWorker::IndepWorker(TraceManager* manager, int id)
{
    done = true;
    retrieveDone = true;
    lostTrack = true;
    willLost = false;
    resumeForRetrieval = false;
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
    static Timer t;
    t.start();
    mat trackedMatches, matchesAll;
    image = img.clone();
    if (lostTrack) {
        if (done) {
            done = false;
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
                if (!extractor.get()) {
                    extractor = ORB::create(4000);
                }
                vector<KeyPoint> rawKeypoints;
                extractor->detect(image, rawKeypoints);
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
                
                for (int i = 0; i < rawKeypoints.size(); i++) {
                    if (delegate->inConvex(rawKeypoints[i].pt) == -1) {
                        if (inputKeypoints.size() < inputKeypointSize) {
                            inputKeypoints.push_back(rawKeypoints[i]);
                        }
                    }
                }
                extractor->compute(image, inputKeypoints, inputDescriptors);
                
                notTrackedMatches = getMatches(inputKeypoints, inputDescriptors);
                
                reconstruction->ReconstructPlanarUnconstrIter2D( notTrackedMatches, resMesh, inlierMatchIdxs );
                
                if (inlierMatchIdxs.n_rows > THRESHOLD) {
                    lostTrack = false;
                    willLost = false;
                    initMatches = (int)inlierMatchIdxs.n_rows;
                }
                done = true;
            });
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (!lostTrack) {
        // if there are lots of points, random track them
//        if (matchesInlier.n_rows > 150) {
//            arma::vec idx(150);
//            idx.randu();
//            idx *= matchesInlier.n_rows - 1;
//            arma::uvec uidx = conv_to<uvec>::from(idx);
//            matchesInlier = matchesInlier.rows(uidx);
//        }
        
        {
            std::unique_lock<std::mutex> lock(delegate->shared_mutex);
            trackedMatches = tracker->TrackMatches(img, matchesInlier);
        }
//        if (done && !cachedMatches.is_empty()) {
//            matchesAll = join_vert(cachedMatches, trackedMatches);
        if (crtFrame == 0) {
            matchesAll = notTrackedMatches;
        } else {
            matchesAll = trackedMatches;
        }
        if (retrieveDone && crtFrame % retrieveInterval == 1) {
            retrieveDone = false;
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
                vector<cv::KeyPoint> __inputKeypoints;
                cv::Mat __inputDescriptors;
                
                cv::Mat __img = img.clone();
                vector<KeyPoint> rawKeypoints;
                extractor->detect(__img, rawKeypoints);
                for (int i = 0; i < rawKeypoints.size(); ++i) {
                    const Point2d& pt = rawKeypoints[i].pt;
                    if (delegate->getHull(id).isInConvexHull(pt) && __inputKeypoints.size() < inputKeypointSize) {
                        __inputKeypoints.push_back(rawKeypoints[i]);
                    }
                }
                extractor->compute(__img, __inputKeypoints, __inputDescriptors);
                cachedMatches = getMatches(__inputKeypoints, __inputDescriptors);
                
                retrieveDone = true;
                resumeForRetrieval = true;
            });
        }
        if (resumeForRetrieval && !cachedMatches.is_empty()) {
            resumeForRetrieval = false; 
            matchesAll = join_vert(matchesAll, cachedMatches);
        }
        reconstruction->ReconstructPlanarUnconstrIter2D( matchesAll, resMesh, inlierMatchIdxs );

//        if (!reconstruction->ReconstructPlanarUnconstrIter2D( matchesAll, resMesh, inlierMatchIdxs )) {
//            
//            lostTrack = true;
//            static ConvexHull convex;
//            delegate->setHull(id, convex);
//            cout << "lost " << crtFrame << endl;
//            
//            crtFrame = -1;
//        }
        
        matchesInlier = matchesAll.rows(inlierMatchIdxs);
//        cout << inlierMatchIdxs.n_rows << endl;
        if (inlierMatchIdxs.n_rows < 0.6 * 120) {
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
            Timer tt;
            tt.start();
            reconstruction->ReconstructPlanarUnconstrOnce(matchesInlier, resMesh);
            tt.stop();
//            cout << "Planar once costs " << tt.getElapsedTimeInMilliSec() << " ms\n";
            
            if (Configuration::enableDeformTracking) {
                const arma::mat& ctrlVertices = resMesh.GetCtrlVertices();
                vec cInit = reshape( ctrlVertices, resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..
                reconstruction->ReconstructSoftConstr(cInit, resMesh);
            } else {
                const arma::mat& ctrlVertices = resMesh.GetCtrlVertices();
                vec cInit = reshape( ctrlVertices, resMesh.GetNCtrlPoints()*3, 1 );
                reconstruction->ReconstructEqualConstr(cInit, resMesh);
            }
            
            ConvexHull hull;
            static uvec bound_id;
            if (Configuration::enableDeformTracking) {
                bound_id = {0, 4, 24, 20};
            } else {
                bound_id = {0, 1, 2, 3};
            }
            const arma::mat& boundary = resMesh.GetCtrlVertices().rows(bound_id);
            arma::mat pt2d = delegate->getRealCamera()->ProjectPoints(boundary);
            
            //        cout << "Worker " << id << endl;
            for (int i = 0; i < pt2d.n_rows; i++) {
                Point2d cvpt2d = Point2d(pt2d.at(i, 0), pt2d.at(i, 1));
                hull.addPoint(cvpt2d);
                
                // one of the vertex is out of image
                if (cvpt2d.x > 640 || cvpt2d.x < 0 || cvpt2d.y > 480 || cvpt2d.y < 0) {
                    if (!willLost) {
                        willLost = true;
                    }
                }
            }
            
            std::unique_lock<std::mutex> lock(delegate->shared_mutex);
            Visualization::DrawProjectedMesh(delegate->outputImg, resMesh, *delegate->getRealCamera(), Scalar(0, 0, 255));
//            if (!willLost)
//            {
//                
//            }
            delegate->setHull(id, hull);
        }
        crtFrame++;
    }
//    if (willLost && !delegate->relativePose.empty()) {
//        transformMesh(resMesh);
//        {
//            std::unique_lock<std::mutex> lock(delegate->shared_mutex);
//            Visualization::DrawProjectedMesh(delegate->outputImg, resMesh, *delegate->getRealCamera(), Scalar(0, 0, 255));
//        }
//
//    }
    t.stop();
//    cout << "Trace costs " << t.getElapsedTimeInMilliSec() << " ms\n";
}

void IndepWorker::transformMesh(LaplacianMesh& triangleMesh)
{
    cv::Mat relativePoseT = delegate->relativePose.t();
    relativePoseT.convertTo(relativePoseT, CV_64F);
    mat relativePoseArma( reinterpret_cast<double*>(relativePoseT.data), relativePoseT.cols, relativePoseT.rows );
    mat vertexPoints = triangleMesh.GetVertexCoords();
    
    for (int i = 0; i < vertexPoints.n_rows; i++) {
        mat point(4, 1);
        point(0) = vertexPoints(i, 0);
        point(1) = vertexPoints(i, 1);
        point(2) = vertexPoints(i, 2);
        point(3) = 1.;
        mat transformed = relativePoseArma * point;
        vertexPoints.row(i) = transformed.t();
    }
}

