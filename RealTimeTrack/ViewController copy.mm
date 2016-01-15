//
//  ViewController.m
//  RealTimeTrack
//
//  Created by Neil on 11/24/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#import "ViewController.h"
#import "Data.hpp"
#import "LaplacianMesh.h"
#import "InlierMatchTracker.h"
#import "Reconstruction.h"
#import "Camera.h"
#import "DefinedMacros.h"
#import "Timer.hpp"
#import "Visualization.h"
#import "ImageChoose.hpp"
#import "BriskKeypointMatcher3D2D.hpp"
#import <opencv2/imgcodecs/ios.h>
#import <armadillo/armadillo>
#import <opencv2/videoio/cap_ios.h>


#define TRACK_MODE 1
#define TAKE_MODE 0

static const unsigned int DETECT_THRES = 100;
//using namespace arma;
@interface ViewController () <CvVideoCameraDelegate>{
    int mode;
    bool lostTrack;
    int lastnMatches, frames;
    Timer timer;
    	// Threshold on number of inliers to detect the surface
    
    vector<cv::KeyPoint> refKeypoints, inputKeypoints;
    cv::Mat  refImg, inputImg, perFrameImg;					// Reference image and input image
    cv::Mat referenceImgRGB;
    
    CvVideoCamera* videoCamera;
    arma::urowvec  				ctrPointIds;			// Set of control points
    LaplacianMesh 				*refMesh, resMesh;		// Select planer or non-planer reference mesh
    
    BriskKeypointMatcher3D2D*       keypointMatcher;
    InlierMatchTracker			  matchTracker;
    Reconstruction* 			    reconstruction;
    ImageChoose*                imageChoose;
    
    arma::uvec					inlierMatchIdxs;
    arma::mat   				matchesAll, matchesInlier, matchesRaw;
    
    Camera  modelWorldCamera, modelCamCamera, realCamera;
}
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UIImageView *templateView;

@end

@implementation ViewController
- (IBAction)toggleMode:(id)sender {
    mode = TRACK_MODE;
    lostTrack = true;
    frames = 0;
}

- (void)initSharedVariables {
    mode = 0;
    lastnMatches = 0;
    matchesAll.resize(0, 8);
    ctrPointIds = Data::loadControlPoints();
    refMesh = new LaplacianMesh();
    imageChoose = new ImageChoose();
}

- (void)loadCamerasAndRefMesh {
    modelWorldCamera = Camera(Data::loadA(), Data::loadRt());
    modelCamCamera = Camera(Data::loadA());
    
    //realCamera = Camera(Data::loadA(), Data::loadRt());
    refMesh->Load();
    refMesh->TransformToCameraCoord(modelWorldCamera);		// Convert the mesh into camera coordinate using world camera
    refMesh->SetCtrlPointIDs(ctrPointIds);
    refMesh->ComputeAPMatrices();
    refMesh->computeFacetNormalsNCentroids();
    
    // Init the recontructed mesh
    resMesh = *refMesh;
}

- (void)loadRefImageAndPointMatcher {
    assert(refImg.rows == 480 && refImg.cols == 640);
    referenceImgRGB = refImg;
    
    mat imCorners = Data::loadCorner();
    
    int pad = 0;
    cv::Point topLeft		(imCorners(0,0) - pad, imCorners(0,1) - pad);
    cv::Point topRight		(imCorners(1,0) + pad, imCorners(1,1) - pad);
    cv::Point bottomRight	(imCorners(2,0) + pad, imCorners(2,1) + pad);
    cv::Point bottomLeft	(imCorners(3,0) - pad, imCorners(3,1) + pad);
    
    // Init point matcher
    if (keypointMatcher) {
        delete keypointMatcher;
        keypointMatcher = nullptr;
    }
    keypointMatcher = new BriskKeypointMatcher3D2D( referenceImgRGB,
                                                        topLeft, topRight, bottomRight, bottomLeft,
                                                        *refMesh, modelCamCamera);
    
    //cv::line(refImg, cv::Point(0, 0), cv::Point(640, 480), KPT_COLOR);
    Visualization::DrawAQuadrangle  ( refImg, topLeft, topRight, bottomRight, bottomLeft, TPL_COLOR );
    Visualization::DrawProjectedMesh( refImg, *refMesh, modelCamCamera, MESH_COLOR );
    //_imageView.image = MatToUIImage(refImg);
}

- (IBAction)takePhoto:(id)sender {
    //imageChoose->StoreDescriptors(perFrameImg);
    refImg = perFrameImg;
    [self loadRefImageAndPointMatcher];
    _templateView.image = MatToUIImage(perFrameImg);
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self initSharedVariables];
    [self loadCamerasAndRefMesh];
    
    int 	nUnConstrIters	= 5;
    int 	radiusInit 		  = 5  * pow(Reconstruction::ROBUST_SCALE, nUnConstrIters-1);
    double 	wrInit	  		= 525 * pow(Reconstruction::ROBUST_SCALE, nUnConstrIters-1);
    reconstruction  	= new Reconstruction  ( *refMesh, modelWorldCamera, wrInit, radiusInit, nUnConstrIters );
    reconstruction->SetUseTemporal(true);
    reconstruction->SetUsePrevFrameToInit(true);
    
    videoCamera = [[CvVideoCamera alloc] initWithParentView:_imageView];
    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    videoCamera.delegate = self;
    videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1920x1080;
    videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    videoCamera.defaultFPS = 30;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        [videoCamera start];
    });
        //[self loadRefImageAndPointMatcher];
    //_imageView.image = MatToUIImage(refImg);
    // Do any additional setup after loading the view, typically from a nib.
}

- (void)processImage:(cv::Mat&)image
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(640, 640.0*1920.0/1080.0));
    cv::Rect rect(0, 0, 640, 480);
    
    
    //image = transform;
//    timer.start();
    image = cv::Mat(image, rect);
    
//    cvtColor(image, image, cv::COLOR_RGBA2RGB);
//    cvtColor(image, image, cv::COLOR_RGB2HSV);
//    vector<cv::Mat> channels;
//    split(image, channels);
//    cv::Mat gray = channels[2];
//    cv::Mat thresh, gaussian;
//    adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 25, 20);
//    GaussianBlur(thresh, image, cv::Size(7, 7), 1);
//    timer.stop();
//    cout << "Gaussian blur and gray convert cost: " << timer.getElapsedTimeInMilliSec() << endl;
    perFrameImg = image;
    
    cv::Mat 	inputImgGray;
    cv::cvtColor( image, inputImgGray, cv::COLOR_BGR2GRAY );
    
    // track
    if (mode == TRACK_MODE) {
        if (lostTrack) {
            
                    }
        
//        cv::Mat 	inputImgGray;
//        cv::Mat 	inputImgRGB;
//        inputImgRGB =  image;
////        inputImgGray = image;
//        cv::cvtColor( inputImgRGB, inputImgGray, cv::COLOR_BGR2GRAY );
        
        // ======== Matches between reference image and input image =======
        if (lostTrack) {
//            timer.start();
//            if (!keypointMatcher) {
//                refImg = perFrameImg;
//                [self loadRefImageAndPointMatcher];
//            }
//            
//            int vote;
//            vote = imageChoose->Vote(image, refKeypoints, inputKeypoints);
//            
//            timer.stop();
//            cout << "vote cost " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
//            
//            if (vote > DETECT_THRES) {
//                cout << "detected---------------------------" << endl;
//                
//                timer.start();
//                // may mistakenly track some points out of the actual image
//                matchesRaw = keypointMatcher->MatchImagesByKeypoints(refKeypoints, inputKeypoints); // TODO: Make the input gray image
//                
//                timer.stop();
//                cout << "Number of 3D-2D Fern matches: " << matchesAll.n_rows << endl;
//                cout << "Total time for matching: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
//            } else {
//                for (int i = 0; i < inputKeypoints.size(); i++) {
//                    cv::Point2d aPoint(inputKeypoints[i].pt.x, inputKeypoints[i].pt.y);		// Convert to opencv Point object
//                    int radius = 2;
//                    
//                    cv::circle(image, aPoint, radius, KPT_COLOR, -1);	// -1 mean filled circle
//                }
//                
//                return;
////                matchesRaw = keypointMatcher->MatchImagesByKeypoints(refKeypoints, inputKeypoints);
//            }
            
            timer.start();
            // may mistakenly track some points out of the actual image
            matchesRaw = keypointMatcher->MatchImages3D2D(inputImgGray); // TODO: Make the input gray image
            
            timer.stop();
            cout << "Number of 3D-2D Fern matches: " << matchesAll.n_rows << endl;
            cout << "Total time for matching: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
        }
        
        
        // ================== Track inlier matches ========================
        // make the last frame's inlier matches to be checked to avoid viaration
        
        // if sudden change between two frames, those inliermatches won't exist any more.
        mat trackedMatches = matchTracker.TrackMatches(inputImgGray, matchesInlier);
        
        if (trackedMatches.n_rows > 0 && !lostTrack) {
            matchesAll = trackedMatches;
        } else {
            matchesAll = join_cols(matchesRaw, trackedMatches);
            // ============== Reconstruction without constraints ==============
            
        }
        
        
        timer.start();
        reconstruction->ReconstructPlanarUnconstrIter( matchesAll, resMesh, inlierMatchIdxs );
        matchesInlier = matchesAll.rows(inlierMatchIdxs);
//        if (trackedMatches.n_rows <= 0 || lostTrack) {
//            // eliminate error matches
//            reconstruction->ReconstructPlanarUnconstrIter( matchesAll, resMesh, inlierMatchIdxs );
//            matchesInlier = matchesAll.rows(inlierMatchIdxs);
//        
//        } else {
//            // in track mode we do not need to reject outliers.
////            reconstruction->ReconstructPlanarUnconstrIter( matchesAll, resMesh, inlierMatchIdxs, 5 );
//            matchesInlier = matchesAll.rows(inlierMatchIdxs);
//        }
        
        
        timer.stop();
        cout << "Number of inliers in unconstrained recontruction: " << inlierMatchIdxs.n_rows << endl;
        cout << "Unconstrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
        

        // ============== Constrained Optimization =======================
        if (inlierMatchIdxs.n_rows > DETECT_THRES && lostTrack)
        {
            lostTrack = false;
            lastnMatches = (int)matchesRaw.n_rows;
            timer.start();
            
            vec cInit = reshape( resMesh.GetCtrlVertices(), resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..
            
            //reconstruction->ReconstructIneqConstr(cInit, resMesh);
            reconstruction->ReconstructSoftConstr(cInit, resMesh);
            
            timer.stop();
            cout << "Constrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms \n\n";
            
            Visualization::DrawProjectedMesh( image, resMesh, modelWorldCamera, MESH_COLOR );
        } else if (inlierMatchIdxs.n_rows >= 0.4 * lastnMatches && !lostTrack) {
            lostTrack = false;
            timer.start();

            vec cInit = reshape( resMesh.GetCtrlVertices(), resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..

            //reconstruction->ReconstructIneqConstr(cInit, resMesh);
            reconstruction->ReconstructSoftConstr(cInit, resMesh);

            timer.stop();
            cout << "Constrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms \n\n";
            
            Visualization::DrawProjectedMesh( image, resMesh, modelWorldCamera, MESH_COLOR );
        } else {
            lostTrack = true;
        }
        
        // Visualization
//        if (inlierMatchIdxs.n_rows > DETECT_THRES)
//        {
//            Visualization::DrawProjectedMesh( image, resMesh, modelCamCamera, MESH_COLOR );
//        }
        //if (isDisplayPoints) {
        //Visualization::DrawVectorOfPointsAsPlus(inputImg, matchTracker.GetGoodTrackedPoints(), INLIER_KPT_COLOR);
        assert(image.data != NULL);
        Visualization::DrawPointsAsDot( image, matchesAll.cols(6,7), KPT_COLOR );				  // All points
        Visualization::DrawPointsAsDot( image, matchesInlier.cols(6,7), INLIER_KPT_COLOR );// Inlier points

    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
