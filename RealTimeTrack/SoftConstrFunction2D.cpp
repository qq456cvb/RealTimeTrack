//
//  SoftConstrFunction2D.cpp
//  RealTimeTrack
//
//  Created by Neil on 5/19/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#include "SoftConstrFunction2D.h"

void SoftConstrFunction2D::Evaluate(const arma::vec &x) {
    // Vertex coordinates: First convert control points into vertices
    // verCoords has the form of [x1 x2, x3... y1 y2 y3... z1 z2 z3]
    // We will compute Jacobian w.r.t vertex variables first, then we compute
    // Jacobian w.r.t actual variables using composition rules
    vec vertCoords = this->P * x;
    vec CPx;
    CPx.set_size(this->refEdgeLens.n_elem);
    
    int nVertVars	= this->P.n_rows;				// Number of vertex variables = 3 * #vertices
    int nVerts		= nVertVars / 3;				// Number of vertices
    int nEdges		= this->refEdgeLens.n_elem;		// Number of edges (#constraints)
    
    // TODO: Make Jv to be attribute of class to avoid re-allocating each evaluation
    // Jacobian w.r.t all vertex coordinates. Then this->J = Jv * P since x = P*c
    mat Jv = zeros(nEdges, nVertVars);
    
    // Iterate through all edges to compute function value and derivatives
    for (int i = 0; i < nEdges; i++)
    {
        // An edge has two vertices
        int vertID1 = edges(0, i);
        int vertID2 = edges(1, i);
        
        // Indices of x,y,z coordinates in the vector vertCoords [x1 x2, x3... y1 y2 y3... z1 z2 z3]
        int x1Idx	= vertID1;
        int y1Idx	= x1Idx + nVerts;
        int z1Idx	= y1Idx + nVerts;
        
        int x2Idx	= vertID2;
        int y2Idx	= x2Idx + nVerts;
        int z2Idx	= y2Idx + nVerts;
        
        // Coordinates of two vertices (x1,y1,z1) & (x2,y2,z2)
        double x1	= vertCoords(x1Idx);
        double y1	= vertCoords(y1Idx);
        double z1	= vertCoords(z1Idx);
        
        double x2	= vertCoords(x2Idx);
        double y2	= vertCoords(y2Idx);
        double z2	= vertCoords(z2Idx);
        
        // Edge length
        double edgeLen = sqrt( pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2) );
        
        // Compute function value = "edge length" - "reference edge length"
        // 有一些存在len大于refLen的边，这不符合实际情况，因此要做constrained optimization
        CPx(i) = edgeLen - refEdgeLens(i);
        // if (edgeLen > refEdgeLens(i))
        // {
        // 	std::cout << "FOUND ONE" << std::endl;
        // }
        
        // Compute derivatives. Recall the derivative: sqrt'(x) = 1/(2*sqrt(x))
        Jv(i, x1Idx) = (x1-x2) / edgeLen;
        Jv(i, y1Idx) = (y1-y2) / edgeLen;
        Jv(i, z1Idx) = (z1-z2) / edgeLen;
        
        Jv(i, x2Idx) = (x2-x1) / edgeLen;
        Jv(i, y2Idx) = (y2-y1) / edgeLen;
        Jv(i, z2Idx) = (z2-z1) / edgeLen;
    }
    
    this->F = BPwAP * x;
    this->C = CPx;
    // Jacobian w.r.t actual variables using composition chain rule
    //    mat m1 = arma::join_cols(MPwAP, Jv * P);
    //mat m2 = arma::join_rows(2*trans(MPwAP*x), 2 * trans(CPx));
    this->J = BPwAP;
    this->A = Jv * P;
    
    //cout << this->J << endl;
    vec CPxabs = arma::abs(CPx);
    
#if 0
    cout << "Constraint function mean: " << arma::mean(CPxabs) << endl;
#endif
}