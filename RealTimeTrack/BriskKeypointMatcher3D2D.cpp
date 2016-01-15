//
//  BriskKeypointMatcher3D2D.cpp
//  RealTimeTrack
//
//  Created by Neil on 12/2/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#include "BriskKeypointMatcher3D2D.hpp"

BriskKeypointMatcher3D2D::BriskKeypointMatcher3D2D(const cv::Mat referenceImgRGB,
                                                   const cv::Point& topLeft,    const cv::Point& topRight,
                                                   const cv::Point& bottomRight, const cv::Point& bottomLeft,
                                                   const TriangleMesh& referenceMesh, const Camera& camCamera) : KeypointMatcher3D2D(referenceImgRGB, topLeft, topRight, bottomRight, bottomLeft, referenceMesh, camCamera)
{
    if (!detector.get()) {
        detector = cv::BRISK::create();
    }
    if (!computor.get()) {
        computor = cv::BRISK::create();
    }
//    std::vector<cv::KeyPoint> objectKeypoints;
//    detector->detect(this->templateImageGray, objectKeypoints);
//    //    std::nth_element(objectKeypoints.begin(), objectKeypoints.begin()+refDescriptorSize, objectKeypoints.end(), [](cv::KeyPoint p1, cv::KeyPoint p2) {
//    //        return (p2.response < p1.response);
//    //    });
//    std::sort(objectKeypoints.begin(), objectKeypoints.end(), [](cv::KeyPoint p1, cv::KeyPoint p2) {
//        return (p2.response < p1.response);
//    });
//    objectKeypoints = std::vector<cv::KeyPoint>(objectKeypoints.begin(), objectKeypoints.begin()+refDescriptorSize);
//    computor->compute(this->templateImageGray, objectKeypoints, objectDescriptors);
//    std::cout << "there are " << objectDescriptors.rows << " descriptors." << std::endl;
//    
//    // Pre-compute 3D coordinates of model keypoints
//    existed3DRefKeypoints.resize(objectDescriptors.rows);
//    bary3DRefKeypoints.set_size(objectDescriptors.rows, 6);
//    
//    for(int i = 0; i < objectDescriptors.rows; i++)
//    {
//        cv::Point2d refPoint( objectKeypoints[i].pt.x, objectKeypoints[i].pt.y );
//        arma::rowvec intersectPoint;
//        bool isIntersect = this->find3DPointOnMesh(refPoint, intersectPoint);
//        
//        existed3DRefKeypoints[i] = isIntersect;
//        if (isIntersect)
//        {
//            this->bary3DRefKeypoints(i, arma::span(0,5)) = intersectPoint;
//        }
//    }
}

BriskKeypointMatcher3D2D::~BriskKeypointMatcher3D2D()
{
    delete detector;
    delete computor;
}

int BriskKeypointMatcher3D2D::matchImages2D2D(cv::Mat inputImageGray)
{
    //    mat matchesSort(detector->number_of_model_points, 5);
    //
    //    detector->detect(inputImageGray);
    //    int number_of_matches = 0;
    //    for(int i = 0; i < detector->number_of_model_points; i++)
    //    {
    //        if (detector->model_points[i].class_score > 0)
    //        {
    //            match1Data[2*number_of_matches]   = detector->model_points[i].fr_u();
    //            match1Data[2*number_of_matches + 1]  = detector->model_points[i].fr_v();
    //
    //            match2Data[2*number_of_matches]    = detector->model_points[i].potential_correspondent->fr_u();
    //            match2Data[2*number_of_matches + 1]  = detector->model_points[i].potential_correspondent->fr_v();
    //
    //            matchesSort(number_of_matches, 0)  = detector->model_points[i].class_score;
    //            matchesSort(number_of_matches, 1)  = detector->model_points[i].fr_u();
    //            matchesSort(number_of_matches, 2)  = detector->model_points[i].fr_v();
    //            matchesSort(number_of_matches, 3)  = detector->model_points[i].potential_correspondent->fr_u();
    //            matchesSort(number_of_matches, 4)  = detector->model_points[i].potential_correspondent->fr_v();
    //
    //            number_of_matches++;
    //        }
    //    }
    //
    //    // Sort matches in decreasing order of score
    //    matchesSort.resize(number_of_matches, 5);
    //    uvec indices = sort_index( matchesSort.col(0), 1 );
    //
    //    for (int i = 0; i < number_of_matches; ++i)
    //    {
    //        match1Data[2*number_of_matches]   = matchesSort(indices(i), 1);
    //        match1Data[2*number_of_matches + 1]  = matchesSort(indices(i), 2);
    //        match2Data[2*number_of_matches]    = matchesSort(indices(i), 3);
    //        match2Data[2*number_of_matches + 1]  = matchesSort(indices(i), 4);
    //    }
    //
    //    return number_of_matches;
    return 0;
}

arma::mat BriskKeypointMatcher3D2D::MatchImagesByKeypoints(const vector<cv::KeyPoint> &referenceKeypoints, const vector<cv::KeyPoint> &inputKeypoints, const arma::mat& ref3DPoints)
{
    existed3DRefKeypoints.resize(referenceKeypoints.size());
    bary3DRefKeypoints.set_size(referenceKeypoints.size(), 6);
    matches3D2D.clear();
    matches3D2D.resize(referenceKeypoints.size(), 9);
    
    timer.start();
    for(int i = 0; i < referenceKeypoints.size(); i++)
    {
            matches3D2D(i, arma::span(0,5)) = ref3DPoints.row(i);
            matches3D2D(i, 6) = inputKeypoints[i].pt.x;
            matches3D2D(i, 7) = inputKeypoints[i].pt.y;
            
            // ID of the 3D point of the match
            matches3D2D(i, 8) = i;
    }
    timer.stop();
    cout << "find 3D point cost " << timer.getElapsedTimeInMilliSec() << endl;
    if (referenceKeypoints.size() <= 0) {
        return arma::mat(0, matches3D2D.n_cols);
    }
    return matches3D2D.rows(0, referenceKeypoints.size()-1);
}

arma::mat BriskKeypointMatcher3D2D::MatchImages3D2D(const cv::Mat& inputImageGray)
{
    // Convert color to gray
    //    assert(inputImageRGB.channels() == 3);
    //    cv::Mat inputImageGray;
    //    cv::cvtColor(inputImageRGB, inputImageGray, cv::COLOR_BGR2GRAY);
    
#if 0
    // ================ For visualization of matches only ================
    int n2D2DMatches = this->matchImages2D2D(inputImageGray);    // Result is stored in match1Data, match2Data
    arma::mat matches2D2D(n2D2DMatches, 4);
    for (int i = 0; i < n2D2DMatches; i++)
    {
        matches2D2D(i, 0) = match1Data[2*i];
        matches2D2D(i, 1) = match1Data[2*i+1];
        
        matches2D2D(i, 2) = match2Data[2*i];
        matches2D2D(i, 3) = match2Data[2*i+1];
    }
    Visualization::VisualizeMatches(this->referenceImgRGB, inputImageRGB, matches2D2D, 10);
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#endif
    
    // Do matching
    std::vector<cv::KeyPoint> sceneKeypoints;
    cv::Mat sceneDescriptors;
    cv::Mat results;
    cv::Mat dists;
    
    timer.start();
    detector->detect(inputImageGray, sceneKeypoints);
    timer.stop();
    std::cout << "detect cost " << timer.getElapsedTimeInMilliSec() << " ms" << std::endl;
    timer.start();
    std::sort(sceneKeypoints.begin(), sceneKeypoints.end(), [](cv::KeyPoint p1, cv::KeyPoint p2) {
        return (p2.response < p1.response);
    });
    //    std::nth_element(sceneKeypoints.begin(), sceneKeypoints.begin()+inputDescriptorSize, sceneKeypoints.end(), [](cv::KeyPoint p1, cv::KeyPoint p2) {
    //        return (p2.response < p1.response);
    //    });
    sceneKeypoints = std::vector<cv::KeyPoint>(sceneKeypoints.begin(), sceneKeypoints.begin()+inputDescriptorSize);
    computor->compute(inputImageGray, sceneKeypoints, sceneDescriptors);
    timer.stop();
    std::cout << "compute cost " << timer.getElapsedTimeInMilliSec() << " ms" << std::endl;
    
    timer.start();
    static cv::flann::Index flannIndex(objectDescriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
    flannIndex.knnSearch(sceneDescriptors, results, dists, 2, cv::flann::SearchParams());
    
    
    // NDDR method
    int nMatches = 0;
    float nndrRatio = 0.8f;
    for (int i = 0; i < sceneDescriptors.rows; ++i)
    {
        //std::cout << dists.at<int>(i, 0) << std::endl;
        if(results.at<int>(i,0) >= 0 && results.at<int>(i,1) >= 0 &&
           dists.at<int>(i,0) <= nndrRatio * dists.at<int>(i,1)) { // good match
            int index = results.at<int>(i,0);
            if (existed3DRefKeypoints[index])
            {
                matches3D2D(nMatches, arma::span(0,5)) = this->bary3DRefKeypoints.row(index);
                
                matches3D2D(nMatches, 6) = sceneKeypoints[i].pt.x;
                matches3D2D(nMatches, 7) = sceneKeypoints[i].pt.y;
                
                // ID of the 3D point of the match
                matches3D2D(nMatches, 8) = index;
                nMatches++;
            }
        }
    }
    
    timer.stop();
    std::cout << "NDDR search by flann index cost " << timer.getElapsedTimeInMilliSec() << " ms" << std::endl;
    std::cout << "nMatches size: " << nMatches << std::endl;
    
    if (nMatches > 0)
        return matches3D2D.rows(0, nMatches-1);
    else
        return arma::mat(0, matches3D2D.n_cols);
}
