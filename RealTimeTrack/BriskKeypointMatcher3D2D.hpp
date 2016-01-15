//
//  BriskKeypointMatcher3D2D.hpp
//  RealTimeTrack
//
//  Created by Neil on 12/2/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#ifndef BriskKeypointMatcher3D2D_hpp
#define BriskKeypointMatcher3D2D_hpp

#include "KeypointMatcher3D2D.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Timer.hpp"
#include <opencv2/features2d/features2d.hpp>

using namespace std;
class BriskKeypointMatcher3D2D : public KeypointMatcher3D2D
{
private:
    Timer timer;
    cv::Mat objectDescriptors;
    cv::Ptr<cv::BRISK> detector;
    cv::Ptr<cv::BRISK> computor;
    const int refDescriptorSize = 500, inputDescriptorSize = 500;
    // Precomputed 3D bary-centric coordinates of all keypoints in reference image
    arma::mat        bary3DRefKeypoints;
    vector<bool>     existed3DRefKeypoints;    // To indicate a feature points lies on the reference mesh.
    
public:
    // Constructor
    BriskKeypointMatcher3D2D(const cv::Mat referenceImgRGB,
                             const cv::Point& topLeft,    const cv::Point& topRight,
                             const cv::Point& bottomRight, const cv::Point& bottomLeft,
                             const TriangleMesh& referenceMesh, const Camera& camCamera);
    
    // Destructor
    virtual ~BriskKeypointMatcher3D2D();
    
    // Override the virtual function of the parent class. Compute 3D-2D matches directly.
    // Don't use 3D-2D match function of the parent class to speed up
    virtual arma::mat MatchImages3D2D(const cv::Mat& inputImageGray) override;
    
    virtual arma::mat MatchImagesByKeypoints(const vector<cv::KeyPoint>& referenceKeypoints, const vector<cv::KeyPoint>& inputKeypoints, const arma::mat& ref3DPoints);
    
private:
    // Override the PURE virtual function defined in the base class
    virtual int matchImages2D2D(cv::Mat inputImageGray);
};
#endif /* BriskKeypointMatcher3D2D_hpp */
