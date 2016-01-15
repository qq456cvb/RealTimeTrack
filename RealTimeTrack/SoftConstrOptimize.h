//
//  SoftConstrOptimize.hpp
//  RealTimeTrack
//
//  Created by Neil on 12/14/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#ifndef SoftConstrOptimize_h
#define SoftConstrOptimize_h

#include <stdio.h>
#include <armadillo/armadillo>
#include "SoftConstrFunction.h"
#include "LaplacianMesh.h"

using namespace arma;

class SoftConstrOptimize {
protected:
    // ---------- Parameters used by optimization algorithm ----------------
    int 			nIters;			// Number of iterations
    int lambda;
    const LaplacianMesh& refMesh;
public:
    
    // Constructor
    SoftConstrOptimize (int nIters, const LaplacianMesh& pRefMesh) : refMesh (pRefMesh)
    {
        this->nIters = nIters;
        lambda = 5000;
    }
    
    virtual ~SoftConstrOptimize() {}
    
    void SetNIterations(int nIters) {
        this->nIters = nIters;
    }
    
    virtual arma::vec OptimizeLagrange(const arma::vec& xInit, SoftConstrFunction& objtFunction, LaplacianMesh& resMesh);
    
private:
    void takeAStepLagrange(arma::vec& x, SoftConstrFunction& objtFunction, LaplacianMesh& resMesh);
};

#endif /* SoftConstrOptimize_hpp */
