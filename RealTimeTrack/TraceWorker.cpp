//
//  TraceWorker.cpp
//  RealTimeTrack
//
//  Created by Neil on 5/5/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "TraceWorker.hpp"
#include "TraceManager.hpp"
#include "Configuration.hpp"

TraceWorker::TraceWorker(TraceManager* manager, int id)
{
    this->delegate = manager;
    this->id = id;
    reconstruction      = new Reconstruction  ( *delegate->getRefMesh(), *delegate->getRealCamera(), wrInit, radiusInit, nUnConstrIters );
    reconstruction->SetUseTemporal(true);
    reconstruction->SetUsePrevFrameToInit(false);
    this->status = AVAILABLE;
}

TraceWorker::~TraceWorker()
{
    if (this->reconstruction) {
        delete this->reconstruction;
    }
    if (this->tracker) {
        delete this->tracker;
    }
}

void TraceWorker::reset(const LaplacianMesh* refMesh, const arma::mat& refKeypoints3d, const vector<cv::KeyPoint>& refKeypoints2d, const cv::Mat& descriptors)
{
    if (this->status == AVAILABLE) {
        resMesh = *refMesh;
        this->tracker = new InlierMatchTracker();
        this->refKeypoints = &refKeypoints2d;
        this->refDescriptors = &descriptors;
        this->bary3DRefKeypoints = &refKeypoints3d;
        this->status = BUSY;
        crtFrame = 0;
        
        // TODO: make every worker response to corresponding candidate to avoid realloc flann index
        
        matchesInlier.clear();
        inlierMatchIdxs.clear();
    } else {
        cerr << "Unexpected status for worker";
        assert(false);
    }
    
}

void TraceWorker::stop()
{
    if (this->status == BUSY) {
        if (this->tracker) {
            delete this->tracker;
            this->tracker = nullptr;
        }
        
        this->status = AVAILABLE;
    } else {
        cerr << "Unexpected status for worker";
        assert(false);
    }
    
}

arma::mat TraceWorker::getMatches(const vector<cv::KeyPoint>& crtKeypoints, const cv::Mat& crtDescriptors)
{
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
//        rowvec vids = matches3D2D(i, arma::span(0, 2));
//        rowvec barys = matches3D2D(i, arma::span(3, 5));
//        rowvec point3D = barys(0) * allPoints.row(vids(0)) + barys(1) * allPoints.row(vids(1)) + barys(2) * allPoints.row(vids(2));
//        auto point2D = delegate->imageDatabase->getRefCamera().ProjectAPoint(point3D.t());
//        cout << point2D << endl;
        matches3D2D(i, 6) = inputKeypoints[i].pt.x;
        matches3D2D(i, 7) = inputKeypoints[i].pt.y;
//        cout << inputKeypoints[i].pt.x << " " << inputKeypoints[i].pt.y << endl;
        // ID of the 3D point of the match
        matches3D2D(i, 8) = i;
    }
    if (referenceKeypoints.size() <= 0) {
        return arma::mat(0, matches3D2D.n_cols);
    }
    return matches3D2D.rows(0, referenceKeypoints.size()-1);
}
/**
 *  trace with current frames keypoints
 *
 *  @param img            current image
 *  @param crtKeypoints   current keypoints
 *  @param crtDescriptors current descriptors
 */
void TraceWorker::trace(cv::Mat &img, const vector<cv::KeyPoint>& crtKeypoints, const cv::Mat& crtDescriptors)
{
    static Timer timer;
    timer.start();
    assert(this->status == BUSY);
    
    mat trackedMatches, notTrackedMatches;
    {
        std::unique_lock<std::mutex> lock(delegate->shared_mutex);
        trackedMatches = tracker->TrackMatches(img, matchesInlier);
    }
    timer.stop();
    
    cout << "track match cost " << timer.getElapsedTimeInMilliSec() << endl;
    arma::mat matchesAll;
    if (crtFrame % detectInterval == 0) { // recompute
        vector<KeyPoint> validKeypoints;
        validKeypoints.reserve(crtKeypoints.size());
        cv::Mat validDescriptors(0, crtDescriptors.cols, crtDescriptors.type());
        if (crtFrame == 0) {
            // init
            for (int i = 0; i < crtKeypoints.size(); ++i) {
                const Point2d& pt = crtKeypoints[i].pt;
                if (delegate->inConvex(pt)) {
                    continue;
                }
                validKeypoints.push_back(crtKeypoints[i]);
                vconcat(validDescriptors, crtDescriptors.row(i), validDescriptors);
            }
        } else {
            // only track its own region
            for (int i = 0; i < crtKeypoints.size(); ++i) {
                const Point2d& pt = crtKeypoints[i].pt;
                if (delegate->getHull(id).isInConvexHull(pt)) {
                    validKeypoints.push_back(crtKeypoints[i]);
                    vconcat(validDescriptors, crtDescriptors.row(i), validDescriptors);
                }
            }
        }
        notTrackedMatches = getMatches(validKeypoints, validDescriptors);
        matchesAll = join_vert(notTrackedMatches, trackedMatches);
    } else { // track mode
        matchesAll = trackedMatches;
    }
//    
    timer.start();
    reconstruction->ReconstructPlanarUnconstrIter2D( matchesAll, resMesh, inlierMatchIdxs );
    matchesInlier = matchesAll.rows(inlierMatchIdxs);
    {
        std::unique_lock<std::mutex> lock(delegate->shared_mutex);
        Visualization::DrawPointsAsDot( delegate->outputImg, matchesAll.cols(6,7), KPT_COLOR );				  // All points
        Visualization::DrawPointsAsDot( delegate->outputImg, matchesInlier.cols(6,7), INLIER_KPT_COLOR );// Inlier points
    }
    timer.stop();
    cout << "2d uncstr cost " << timer.getElapsedTimeInMilliSec() << endl;

    timer.start();
    reconstruction->ReconstructPlanarUnconstrOnce(matchesInlier, resMesh);
//    matchesInlier = matchesInlier.rows(inlierMatchIdxs);
    timer.stop();
//    
//    timer.start();
//    reconstruction->ReconstructPlanarUnconstrIter(matchesAll, resMesh, inlierMatchIdxs);
//    matchesInlier = matchesAll.rows(inlierMatchIdxs);

//    timer.stop();
    
    cout << "inlier: " << inlierMatchIdxs.n_rows << endl;
    cout << "uncstr cost " << timer.getElapsedTimeInMilliSec() << endl;
    
    timer.start();
    if (crtFrame == 0) {
        initMatches = (int)inlierMatchIdxs.n_rows;
    }
    if ((crtFrame == 0 && inlierMatchIdxs.n_rows > THRESHOLD) || (crtFrame != 0 && inlierMatchIdxs.n_rows > 0.6 * initMatches)) {
        
        if (Configuration::enableDeformTracking) {
        
            const arma::mat& ctrlVertices = resMesh.GetCtrlVertices();
            vec cInit = reshape( ctrlVertices, resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..
            reconstruction->ReconstructSoftConstr(cInit, resMesh);
        }
        // update the convex hull
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
    } else { // lost
        cout << "Worker " << id << " stop...\n";
        static ConvexHull convex;
        delegate->setHull(id, convex);
        stop();
    }
    crtFrame++;
    
    timer.stop();
    cout << "constr cost " << timer.getElapsedTimeInMilliSec() << " ms\n";
}
