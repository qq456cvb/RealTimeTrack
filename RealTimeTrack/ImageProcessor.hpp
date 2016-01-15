//
//  ImageProcessor.hpp
//  RealTimeTrack
//
//  Created by Neil on 12/4/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#ifndef ImageProcessor_hpp
#define ImageProcessor_hpp

#include <stdio.h>
#import "Data.hpp"
#import "LaplacianMesh.h"
#import "InlierMatchTracker.h"
#import "Reconstruction.h"
#import "Camera.h"
#import "DefinedMacros.h"
#import "Timer.hpp"
#import "Visualization.h"
#import "BriskKeypointMatcher3D2D.hpp"
#import <opencv2/imgcodecs/ios.h>
#import <armadillo/armadillo>
#import <opencv2/videoio/cap_ios.h>

using namespace cv;

class ImageProcessor {
    const int DETECT_THRES = 50;
    Timer timer;
    // Threshold on number of inliers to detect the surface
    
    cv::Mat  refImg, inputImg, perFrameImg;					// Reference image and input image
    //cv::Mat referenceImgRGB;
    
    arma::urowvec  				ctrPointIds;			// Set of control points
    LaplacianMesh 				*refMesh, resMesh;		// Select planer or non-planer reference mesh
    
    BriskKeypointMatcher3D2D*       keypointMatcher;
    InlierMatchTracker			  matchTracker;
    Reconstruction* 			    reconstruction;
    
    arma::uvec					inlierMatchIdxs;
    arma::mat   				matchesAll, matchesInlier;
    
    Camera  modelWorldCamera, modelCamCamera;
    
    void initSharedVariables();
    void loadCamerasAndRefMesh();
    
public:
    void Init();
    void saveTemplate(cv::Mat& img);
    void process(cv::Mat& img);
};


#endif /* ImageProcessor_hpp */
