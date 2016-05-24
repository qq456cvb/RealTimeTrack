//
//  SoftConstrOptimize.cpp
//  RealTimeTrack
//
//  Created by Neil on 12/14/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#include "SoftConstrOptimize.h"
#include "LinearAlgebraUtils.h"
#include "Timer.hpp"

vec SoftConstrOptimize::OptimizeLagrange(const arma::vec& xInit, SoftConstrFunction& objtFunction, LaplacianMesh& resMesh) {
    
    vec x = xInit;						// Variables to be found
    for (int i = 0; i < nIters; i++)
    {
        // Update x and s step by step using LVM algorithm
        SoftConstrOptimize::takeAStepLagrange( x, objtFunction, resMesh );
    }
    
    return x;
}

vec SoftConstrOptimize::OptimizeLagrange(const arma::vec& xInit, SoftConstrFunction2D& objtFunction, LaplacianMesh& resMesh) {
    
    vec x = xInit;						// Variables to be found
    for (int i = 0; i < nIters; i++)
    {
        // Update x and s step by step using LVM algorithm
        SoftConstrOptimize::takeAStepLagrange( x, objtFunction, resMesh );
    }
    
    return x;
}

void SoftConstrOptimize::takeAStepLagrange(arma::vec& x, SoftConstrFunction2D &objtFunction, LaplacianMesh& resMesh) {
    objtFunction.Evaluate(x);
    
    const vec& F = objtFunction.GetF();		// Function value
    const mat& J = objtFunction.GetJ();		// Jacobian. F and J are reference to function.J
    const vec& C = objtFunction.C;
    const mat& A = objtFunction.A;
    
    long long lambda = 8400 * 8400;
    
    double err = 1e-6;
    
    // I don't know why this also works...
    vec dx = LinearAlgebraUtils::LeastSquareSolve(trans(J)*J+lambda*trans(A)*A, -trans(J)*F-lambda*trans(A)*C, err);
    x = x + dx;
}

void SoftConstrOptimize::takeAStepLagrange(arma::vec& x, SoftConstrFunction &objtFunction, LaplacianMesh& resMesh) {
    objtFunction.Evaluate(x);
    
    const vec& F = objtFunction.GetF();		// Function value
    const mat& J = objtFunction.GetJ();		// Jacobian. F and J are reference to function.J
    const vec& C = objtFunction.C;
    const mat& A = objtFunction.A;
    
    long long lambda = 8400;

    // gradient descent works well.
    vec dx = trans(J)*F + lambda * lambda * trans(A)*C;
    double learningRate = 1e-10;
    x = x - learningRate * dx;
//    const mat& paramMat = refMesh.GetParamMatrix();
//    // --------------- Eigen value decomposition --------------------------
//    mat V;
//    vec s;
//    Timer timer;
//    timer.start();
//    
//    // 此处特征值只是其中一个解，在下面会对这个解进行Scale使得它的边长和ref mesh的边长相等
//    eig_sym(s, V, J);
//    timer.stop();
//    //cout << "Eigen(): " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
//    const vec& c = V.col(0);
//    
//    mat matC = reshape(c, refMesh.GetNCtrlPoints(), 3);
//    
//    // Update vertex coordinates
//    resMesh.SetVertexCoords(paramMat * matC);
//    
//    // Resulting mesh yields a correct projection on image but it does not preserve lengths.
//    // So we need to compute the scale factor and multiply with matC
//    
//    // Determine on which side the mesh lies: -Z or Z
//    double meanZ = mean(resMesh.GetVertexCoords().col(2));
//    int globalSign = meanZ > 0 ? 1 : -1;
//    
//    const vec& resMeshEdgeLens = resMesh.ComputeEdgeLengths();
//    const vec& refMeshEdgeLens = refMesh.GetEdgeLengths();
//    
//    // this seems no difference if do not scale the points?
//    double scale = globalSign * norm(refMeshEdgeLens, 2) / norm(resMeshEdgeLens, 2);
//    
//    //std::cout << scale << std::endl;
//    // Update vertex coordinates
//    resMesh.SetVertexCoords(scale * paramMat * matC);
//    
//    objtFunction.Evaluate(c);
}