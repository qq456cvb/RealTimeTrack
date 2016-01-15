//
//  SoftConstrOptimize.hpp
//  RealTimeTrack
//
//  Created by Neil on 12/14/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#ifndef SoftConstrFunction_h
#define SoftConstrFunction_h

#include <stdio.h>
#include <armadillo/armadillo>
#include "Function.h"
using namespace arma;

class SoftConstrFunction : public Function {
private:
    const arma::mat     &MPwAP;
    const arma::mat		&P;				// Matrix P in which x = P*c. Size of 3*#vertices x 3*#controlPoints
    const arma::umat	&edges;			// Edges, size of 2 * #edges each of which is represented by two vertex ids.
    const arma::vec		&refEdgeLens;	// Edge lengths of the reference mesh, size of #edges x 1
    // Here, we uses const & to avoid deep object copying and get better performance
    
public:
    vec C;
    mat A;
    
    // Constructor that initializes the references. p stands for parameters
    SoftConstrFunction (const arma::mat& pP, const arma::mat& pMPwAP, const arma::umat& pEdges, const arma::vec& pReferencLens) :
    P			( pP ),
    edges		( pEdges ),
    refEdgeLens	( pReferencLens ),
    MPwAP       ( pMPwAP )
    {
        // Initialize variable F and J so that they can be re-used in Evaluate(x)
        //int nEdges		= refEdgeLens.n_elem;
        //int nActualVars = this->P.n_cols;
        
       // this->F.set_size(nEdges);
       // this->J.set_size(nEdges, nActualVars);
    }
    
    
    virtual void Evaluate( const arma::vec& x);
};

#endif /* SoftConstrOptimize_hpp */
