//
//  ImageProcessor.cpp
//  RealTimeTrack
//
//  Created by Neil on 12/4/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#include "ImageProcessor.hpp"

void ImageProcessor::initSharedVariables() {
    ctrPointIds = Data::loadControlPoints();
    refMesh = new LaplacianMesh();
}

void ImageProcessor::loadCamerasAndRefMesh() {
    modelWorldCamera = Camera(Data::loadA(), Data::loadRt());
    modelCamCamera = Camera(Data::loadA());
    
    refMesh->Load();
    refMesh->TransformToCameraCoord(modelWorldCamera);		// Convert the mesh into camera coordinate using world camera
    refMesh->SetCtrlPointIDs(ctrPointIds);
    refMesh->ComputeAPMatrices();
    refMesh->computeFacetNormalsNCentroids();
    
    // Init the recontructed mesh
    resMesh = *refMesh;
}

void ImageProcessor::Init() {
    this->initSharedVariables();
    this->loadCamerasAndRefMesh();
}

void ImageProcessor::saveTemplate(cv::Mat& img) {
    cv::resize(img, img, cv::Size(640, 480));
    //assert(img.rows == 480 && img.cols == 640);
    refImg = img;
    
    mat imCorners = Data::loadCorner();
    
    int pad = 0;
    cv::Point topLeft		(imCorners(0,0) - pad, imCorners(0,1) - pad);
    cv::Point topRight		(imCorners(1,0) + pad, imCorners(1,1) - pad);
    cv::Point bottomRight	(imCorners(2,0) + pad, imCorners(2,1) + pad);
    cv::Point bottomLeft	(imCorners(3,0) - pad, imCorners(3,1) + pad);
    
    // Init point matcher
    if (keypointMatcher) {
        delete keypointMatcher;
    }
    keypointMatcher = new BriskKeypointMatcher3D2D( refImg,
                                                   topLeft, topRight, bottomRight, bottomLeft,
                                                   *refMesh, modelCamCamera);
    
    //cv::line(refImg, cv::Point(0, 0), cv::Point(640, 480), KPT_COLOR);
    Visualization::DrawAQuadrangle  ( refImg, topLeft, topRight, bottomRight, bottomLeft, TPL_COLOR );
    Visualization::DrawProjectedMesh( refImg, *refMesh, modelCamCamera, MESH_COLOR );
}

void ImageProcessor::process(cv::Mat& image) {
    cv::resize(image, image, cv::Size(640,480));
    
    cv::Mat 	inputImgGray;
    cv::Mat 	inputImgRGB;
    inputImgRGB =  image;
    cv::cvtColor( inputImgRGB, inputImgGray, cv::COLOR_BGR2GRAY );
    
    // ======== Matches between reference image and input image =======
    
    timer.start();
    
    // may mistakenly track some points out of the actual image
    matchesAll = keypointMatcher->MatchImages3D2D(inputImgGray); // TODO: Make the input gray image
    
    timer.stop();
    cout << "Number of 3D-2D Fern matches: " << matchesAll.n_rows << endl;
    cout << "Total time for matching: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
    
    // ================== Track inlier matches ========================
    // make the last frame's inlier matches to be checked to avoid viaration
    
    // if sudden change between two frames, those inliermatches won't exist any more.
    mat trackedMatches = matchTracker.TrackMatches(inputImgGray, matchesInlier);
    matchesAll 		   = join_cols(matchesAll, trackedMatches);
    
    // ============== Reconstruction without constraints ==============
    timer.start();
    // eliminate error matches
    reconstruction->ReconstructPlanarUnconstrIter( matchesAll, resMesh, inlierMatchIdxs );
    
    matchesInlier = matchesAll.rows(inlierMatchIdxs);
    
    timer.stop();
    cout << "Number of inliers in unconstrained recontruction: " << inlierMatchIdxs.n_rows << endl;
    cout << "Unconstrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
    
    // ============== Constrained Optimization =======================
    if (inlierMatchIdxs.n_rows > DETECT_THRES)
    {
        timer.start();
        
        vec cInit = reshape( resMesh.GetCtrlVertices(), resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..
        
        reconstruction->ReconstructIneqConstr(cInit, resMesh);
        
        timer.stop();
        cout << "Constrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms \n\n";
    }
    
    // Visualization
    if (inlierMatchIdxs.n_rows > DETECT_THRES)
    {
        Visualization::DrawProjectedMesh( image, resMesh, modelCamCamera, MESH_COLOR );
    }
    //if (isDisplayPoints) {
    //Visualization::DrawVectorOfPointsAsPlus(inputImg, matchTracker.GetGoodTrackedPoints(), INLIER_KPT_COLOR);
    Visualization::DrawPointsAsDot( image, matchesAll.cols(6,7), KPT_COLOR );				  // All points
    Visualization::DrawPointsAsDot( image, matchesInlier.cols(6,7), INLIER_KPT_COLOR );// Inlier points
}
