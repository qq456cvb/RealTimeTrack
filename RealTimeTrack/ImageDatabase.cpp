//
//  ImageDatabase.cpp
//  RealTimeTrack
//
//  Created by Neil on 5/6/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "ImageDatabase.hpp"

ImageDatabase::ImageDatabase(const LaplacianMesh &mesh, const Camera &cam):
    referenceMesh(mesh),
    camCamera(cam)
{
    size = 0;
    accumulates.push_back(0);
}

void ImageDatabase::addImage(const cv::Mat &image)
{
    size++;
    if (!extractor.get()) {
        extractor = cv::ORB::create(refDescriptorSize);
    }
    std::vector<cv::KeyPoint> objectKeypoints;
    extractor->detect(image, objectKeypoints);
    std::vector<cv::KeyPoint> strongestKeyPoints;
    if (objectKeypoints.size() < refDescriptorSize) {
        strongestKeyPoints = objectKeypoints;
        keypoints.push_back(strongestKeyPoints);
    } else {
        std::sort(objectKeypoints.begin(), objectKeypoints.end(), [&](cv::KeyPoint p1, cv::KeyPoint p2) {
            return p1.response > p2.response;
        });
        strongestKeyPoints = std::vector<KeyPoint>(objectKeypoints.begin(), objectKeypoints.begin()+refDescriptorSize);
        keypoints.push_back(strongestKeyPoints);
    }

    accumulates.push_back((int)accumulates[accumulates.size()-1] + (int)strongestKeyPoints.size());
    
    cv::Mat objectDescriptors;
    extractor->compute(image, strongestKeyPoints, objectDescriptors);
    
//    assert(objectDescriptors.isContinuous());
    descriptors.push_back(objectDescriptors);
    
    cout << "keypoint size: " << keypoints[keypoints.size()-1].size() << endl;
    cout << "descriptor size: " << descriptors[descriptors.size()-1].rows << endl;

    // store 3D ref points
    arma::mat ref3DPoints(objectDescriptors.rows, 6),
        ref2DPoints(objectDescriptors.rows, 5);
    for(int i = 0; i < strongestKeyPoints.size(); i++)
    {
        cv::Point2d refPoint( strongestKeyPoints[i].pt.x, strongestKeyPoints[i].pt.y );
        arma::rowvec intersectPoint;
        this->find3DPointOnMesh(refPoint, intersectPoint);
        ref3DPoints(i, arma::span(0,5)) = intersectPoint;
    }
    bary3DRefKeypoints.push_back(ref3DPoints);
}

bool ImageDatabase::find3DPointOnMesh(const cv::Point2d& refPoint, rowvec& intersectionPoint) const
{
    bool found = false;
    vec source = zeros<vec>(3);      // Camera center in camera coordinate
    
    vec homorefPoint;                // Homogeneous reference image point
    homorefPoint << refPoint.x << refPoint.y << 1;
    
    vec destination = solve(camCamera.GetA(), homorefPoint);
    
    const umat& facets = referenceMesh.GetFacets(); // each row: [vid1 vid2 vid3]
    const  mat& vertexCoords = referenceMesh.GetVertexCoords(); // each row: [x y z]
    
    double minDepth = INFINITY;
    
    int nFacets = referenceMesh.GetNFacets();
    for (int i = 0; i < nFacets; i++)
    {
        const urowvec &aFacet  = facets.row(i);          // [vid1 vid2 vid3]
        const mat    &vABC    = vertexCoords.rows(aFacet);    // 3 rows of [x y z]
        
        vec bary = findIntersectionRayTriangle(source, destination, vABC);
        if (bary(0) >= 0 && bary(1) >= 0 && bary(2) >= 0)
        {
            double depth = bary(0)*vABC(0,2) + bary(1)*vABC(1,2) + bary(2)*vABC(2,2);
            if (depth < minDepth)
            {
                minDepth = depth;
                intersectionPoint << aFacet(0) << aFacet(1) << aFacet(2)
                << bary(0) << bary(1) << bary(2) << endr;
                
                found = true;
            }
        }
    }
    
    return found;
}

vec ImageDatabase::findIntersectionRayTriangle(const vec& source, const vec& destination, const mat& vABC) const
{
    vec direction = destination - source;
    
    mat A = join_rows(vABC.t(), -direction); // A = [vABC' -d]
    A     = join_cols(A, join_rows(ones(1, 3), zeros(1,1))); // A = [A; [1 1 1 0]]
    
    vec b = join_cols(source, ones(1,1)); // b = [source; 1]
    
    vec X = solve(A, b); // X = A \ b
    
    return X.subvec(0,2);
}

const std::vector<KeyPoint>& ImageDatabase::getKeypoints(int index) const
{
    if (index >= size) {
        assert(false);
        cerr << "index out of bound" << endl;
    }
    return keypoints[index];
}

const cv::Mat& ImageDatabase::getDescriptors(int index) const
{
    if (index >= size) {
        assert(false);
        cerr << "index out of bound" << endl;
    }
    return descriptors[index];
}

const arma::mat& ImageDatabase::getBary3DRefKeypoints(int index) const
{
    
    if (index >= size) {
        assert(false);
        cerr << "index out of bound" << endl;
    }
    return bary3DRefKeypoints[index];

}

const std::vector<std::vector<KeyPoint>>& ImageDatabase::getAllKeypoints() const
{
    return keypoints;
}

const std::vector<cv::Mat>& ImageDatabase::getAllDescriptors() const
{
    return descriptors;
}

const std::vector<int>& ImageDatabase::getAccumulates() const
{
    return accumulates;
}

const Camera& ImageDatabase::getRefCamera() const
{
    return camCamera;
}

const LaplacianMesh& ImageDatabase::getReferenceMesh() const
{
    return referenceMesh;
}