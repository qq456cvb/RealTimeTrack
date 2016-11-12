//
//  ImageChoose.cpp
//  RealTimeTrack
//
//  Created by Neil on 1/6/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "ImageChoose.hpp"


ImageChoose::ImageChoose(const TriangleMesh& mesh, const Camera& camera) :
    camCamera(camera),
    referenceMesh(mesh)
{
    bary3DRefKeypoints.set_size(0, 6);
}

ImageChoose::~ImageChoose() {
    detector.release();
}

void ImageChoose::StoreDescriptors(const cv::Mat &image) {
    if (!detector.get()) {
        detector = cv::BRISK::create();
    }
    std::vector<cv::KeyPoint> objectKeypoints;
    detector->detect(image, objectKeypoints);
    std::sort(objectKeypoints.begin(), objectKeypoints.end(), [&](cv::KeyPoint p1, cv::KeyPoint p2) {
        return p1.response > p2.response;
    });
    std::vector<cv::KeyPoint> strongestKeyPoints(objectKeypoints.begin(), objectKeypoints.begin()+refDescriptorSize);
    keypoints.push_back(objectKeypoints);
    
    cv::Mat objectDescriptors;
    detector->compute(image, strongestKeyPoints, objectDescriptors);
    
    if (descriptors.rows == 0)
    {
        descriptors = objectDescriptors;
    } else {
        vconcat(descriptors, objectDescriptors, descriptors);
    }
    votes.push_back(0);
    if (sizes.size() > 0)
    {
        sizes.push_back(objectDescriptors.rows + sizes[sizes.size()-1]);
    } else {
        sizes.push_back(objectDescriptors.rows);
    }
    
    // store 3D ref points
    mat ref3DPoints(objectDescriptors.rows, 6);
    for(int i = 0; i < strongestKeyPoints.size(); i++)
    {
        cv::Point2d refPoint( strongestKeyPoints[i].pt.x, strongestKeyPoints[i].pt.y );
        arma::rowvec intersectPoint;
        bool isIntersect = this->find3DPointOnMesh(refPoint, intersectPoint);
        ref3DPoints(i, arma::span(0,5)) = intersectPoint;
    }
    bary3DRefKeypoints = join_vert(bary3DRefKeypoints, ref3DPoints);
    
    std::cout << "there are " << objectDescriptors.rows << " descriptors." << std::endl;
    return;
}


int ImageChoose::Vote(const cv::Mat &image, vector<cv::KeyPoint>& refKeypoints, vector<cv::KeyPoint>& inputKeypoints, mat& ref3DPoints) {
    vector<vector<cv::KeyPoint>> refGroupKeypoints;
    vector<vector<cv::KeyPoint>> inputGroupKeypoints;
    
    std::vector<cv::KeyPoint> sceneKeypoints;
    cv::Mat sceneDescriptors;
    cv::Mat results;
    cv::Mat dists;
    
    assert(detector);
    votes.resize(sizes.size());
    refGroupKeypoints.resize(sizes.size());
    inputGroupKeypoints.resize(sizes.size());
    for (int i = 0; i < votes.size(); i++) {
        votes[i] = 0;
    }
    timer.start();
    detector->detect(image, sceneKeypoints);
    std::sort(sceneKeypoints.begin(), sceneKeypoints.end(), [&](cv::KeyPoint p1, cv::KeyPoint p2) {
        return p1.response > p2.response;
    });
    std::vector<cv::KeyPoint> strongestKeyPoints(sceneKeypoints.begin(), sceneKeypoints.begin()+refDescriptorSize);
    detector->compute(image, strongestKeyPoints, sceneDescriptors);
    timer.stop();
    static cv::flann::Index flannIndex(this->descriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
    flannIndex.knnSearch(sceneDescriptors, results, dists, 2, cv::flann::SearchParams());
    
    // NDDR method
    float nndrRatio = 0.8f;
    int cnt = 0;
    vector<uvec> indices;
    indices.resize(sizes.size());
    for (int i = 0; i < sceneDescriptors.rows; ++i)
    {
        //std::cout << dists.at<int>(i, 0) << std::endl;
        if(results.at<int>(i,0) >= 0 && results.at<int>(i,1) >= 0 &&
           dists.at<int>(i,0) <= nndrRatio * dists.at<int>(i,1)) {
            
            int indice = results.at<int>(i,0);
            if (indice < sizes[0])
            {
                votes[0]++;
                refGroupKeypoints[0].push_back(keypoints[0][indice]);
                inputGroupKeypoints[0].push_back(strongestKeyPoints[i]);
                indices[0].insert_rows(indices[0].n_rows, 1);
                indices[0](indices[0].n_rows-1) = indice;
            } else {
                for (int j = 0; j < sizes.size() - 1; ++j)
                {
                    // std::cout << descSize[j+1] << std::endl;
                    if (indice >= sizes[j] && indice < sizes[j+1])
                    {
                        votes[j+1]++;
                        refGroupKeypoints[j+1].push_back(keypoints[j+1][indice-sizes[j]]);
                        inputGroupKeypoints[j+1].push_back(strongestKeyPoints[i]);
                        indices[j+1].insert_rows(indices[j+1].n_rows, 1);
                        indices[j+1](indices[j+1].n_rows-1) = indice;
                        break;
                    }
                }
            }
        }
    }
    for (int i = 0; i < votes.size(); ++i)
    {
        std::cout << "votes" << i << " is " << votes[i] << std::endl;
    }
    auto max_vote = std::max_element(votes.begin(), votes.end());
    int distance = (int)std::distance(votes.begin(), max_vote);
    refKeypoints = refGroupKeypoints[distance];
    inputKeypoints = inputGroupKeypoints[distance];
    ref3DPoints = bary3DRefKeypoints.rows(indices[distance]);
    cout << ref3DPoints.n_rows << " == " << refKeypoints.size() << endl;
    return *max_vote;
}

const cv::Mat ImageChoose::getDescriptors(int index) {
    assert(index < sizes.size());
    if (index == 0) {
        return descriptors.rowRange(0, sizes[0]);
    }
    return descriptors.rowRange(sizes[index-1], sizes[index]);
}

const vector<cv::KeyPoint> ImageChoose::getKeyPoints(int index) {
    return keypoints[index];
}

bool ImageChoose::find3DPointOnMesh(const cv::Point2d& refPoint, rowvec& intersectionPoint) const
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

vec ImageChoose::findIntersectionRayTriangle(const vec& source, const vec& destination, const mat& vABC) const
{
    vec direction = destination - source;
    
    mat A = join_rows(vABC.t(), -direction); // A = [vABC' -d]
    A     = join_cols(A, join_rows(ones(1, 3), zeros(1,1))); // A = [A; [1 1 1 0]]
    
    vec b = join_cols(source, ones(1,1)); // b = [source; 1]
    
    vec X = solve(A, b); // X = A \ b
    
    return X.subvec(0,2);
}
