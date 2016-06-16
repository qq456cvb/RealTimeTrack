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
#import "TraceManager.hpp"
#import "ImageChoose.hpp"
#import "ImageDatabase.hpp"
#import "BriskKeypointMatcher3D2D.hpp"
#import <opencv2/imgcodecs/ios.h>
#import <armadillo/armadillo>
#import <opencv2/videoio/cap_ios.h>


#define TRACK_MODE 1
#define TAKE_MODE 0

static const unsigned int DETECT_THRES = 100;
//using namespace arma;
@interface ViewController () <CvVideoCameraDelegate>{
    Timer timer;
    int mode;
    	// Threshold on number of inliers to detect the surface
    TraceManager* traceManager;
    
    cv::Mat  refImg, inputImg, perFrameImg, testImg;					// Reference image and input image
    cv::Mat referenceImgRGB;
    
    CvVideoCamera* videoCamera;
    arma::urowvec  				ctrPointIds;			// Set of control points
    LaplacianMesh 				*refMesh, resMesh;		// Select planer or non-planer reference mesh
    
    InlierMatchTracker			  matchTracker;
    
    Camera  modelWorldCamera, modelCamCamera, realWorldCamera, realCamCamera;
}
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UIImageView *templateView;

@end

@implementation ViewController
- (IBAction)toggleMode:(id)sender {
    mode = TRACK_MODE;
//    traceManager->outputImg = testImg.clone();
//    traceManager->feed(testImg);
//    _imageView.image = MatToUIImage(traceManager->outputImg);
}

- (void)initSharedVariables {
    mode = TAKE_MODE;
    ctrPointIds = Data::loadControlPoints();
    refMesh = new LaplacianMesh();
}

- (void)loadCamerasAndRefMesh {
    modelWorldCamera = Camera(Data::loadA(), Data::loadRt());
    modelCamCamera = Camera(Data::loadA());
    
    realWorldCamera = Camera(Data::loadRealA(), Data::loadRealRt());
    realCamCamera = Camera(Data::loadRealA());
    
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
//    if (keypointMatcher) {
//        delete keypointMatcher;
//    }
//    keypointMatcher = new BriskKeypointMatcher3D2D( refImg,
//                                                   topLeft, topRight, bottomRight, bottomLeft,
//                                                   *refMesh, modelCamCamera);
    
    //cv::line(refImg, cv::Point(0, 0), cv::Point(640, 480), KPT_COLOR);
    Visualization::DrawAQuadrangle  ( refImg, topLeft, topRight, bottomRight, bottomLeft, TPL_COLOR );
    Visualization::DrawProjectedMesh( refImg, *refMesh, modelCamCamera, MESH_COLOR );
    Visualization::DrawKeyPoints(refImg, *refMesh, ctrPointIds, modelCamCamera, MATCH_COLOR);
}

- (IBAction)takePhoto:(id)sender {
//    cv::Mat ref;
//    NSString* filePath = [[NSBundle mainBundle]
//                              pathForResource:@"cloud" ofType:@"JPG"];
//    UIImage* resImage = [UIImage imageWithContentsOfFile:filePath];
//    UIImageToMat(resImage, ref, true);
//    ref = cv::imread([filePath UTF8String]);
//
//    ref = ref.t();
//    flip(ref, ref, 1);
//    cv::resize(ref, ref, cv::Size(640, 480));
//    cvtColor(ref, ref, CV_BGR2RGB);
////    ref = ref.t();
////    flip(ref, ref, 0);
//    cv::cvtColor(ref, ref, CV_RGB2GRAY);
////    cv::resize(ref, ref, cv::Size(480, 640));
////    ref = ref.t();
////    flip(ref, ref, 0);
//    
//    testImg = ref;
//    _imageView.image = MatToUIImage(ref);
    TraceManager::imageDatabase->addImage(refImg);
//    Timer t;
//    t.start();
//    for (int i = 0; i < 25; i++) {
//        TraceManager::imageDatabase->addImage(ref);
//    }
//    t.stop();
//    cout << "database cost " << t.getElapsedTimeInMilliSec() << endl;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self initSharedVariables];
    [self loadCamerasAndRefMesh];
    
//    NSString* filePath = [[NSBundle mainBundle]
//                          pathForResource:@"cloud" ofType:@"JPG"];
//    UIImage* resImage = [UIImage imageWithContentsOfFile:filePath];

    TraceManager::imageDatabase = new ImageDatabase(*(refMesh), modelCamCamera);
    traceManager = new TraceManager();
    traceManager->init(refMesh, &realCamCamera);

    videoCamera = [[CvVideoCamera alloc] initWithParentView:_imageView];
    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    videoCamera.delegate = self;
    videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
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
//    cv::resize(image, image, cv::Size(640, 640.0*1920.0/1080.0));
//    cv::Rect rect(0, 0, 640, 480);
//    
//    image = cv::Mat(image, rect);
    cv::cvtColor( image, image, cv::COLOR_BGR2RGB );
    
    traceManager->outputImg = image.clone();
    
    cv::cvtColor( image, image, cv::COLOR_RGB2GRAY );
    
    if (mode == TRACK_MODE) {
        traceManager->feed(image);
    } else {
        refImg = image;
    }
    
    image = traceManager->outputImg;
//    if (!testImg.empty()) {
//        image = testImg;
//    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
