//
//  SoftConstrFunction2D.h
//  RealTimeTrack
//
//  Created by Neil on 5/19/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef SoftConstrFunction2D_h
#define SoftConstrFunction2D_h

#include <stdio.h>
#include <stdio.h>
#include <armadillo/armadillo>
#include "Function.h"
using namespace arma;

class SoftConstrFunction2D : public Function {
private:
    const arma::mat     &BPwAP;
    const arma::mat		&P;				// Matrix P in which x = P*c. Size of 3*#vertices x 3*#controlPoints
    const arma::umat	&edges;			// Edges, size of 2 * #edges each of which is represented by two vertex ids.
    const arma::vec		&refEdgeLens;	// Edge lengths of the reference mesh, size of #edges x 1
    // Here, we uses const & to avoid deep object copying and get better performance
    
public:
    vec C;
    mat A;
    
    // Constructor that initializes the references. p stands for parameters
    SoftConstrFunction2D (const arma::mat& pP, const arma::mat& pBPwAP, const arma::umat& pEdges, const arma::vec& pReferencLens) :
    P			( pP ),
    edges		( pEdges ),
    refEdgeLens	( pReferencLens ),
    BPwAP       ( pBPwAP )
    {
        // Initialize variable F and J so that they can be re-used in Evaluate(x)
        //int nEdges		= refEdgeLens.n_elem;
        //int nActualVars = this->P.n_cols;
        
        // this->F.set_size(nEdges);
        // this->J.set_size(nEdges, nActualVars);
    }
    
    
    virtual void Evaluate( const arma::vec& x);
};

#endif /* SoftConstrFunction2D_h */
