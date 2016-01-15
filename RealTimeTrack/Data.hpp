//
//  Data.hpp
//  RealTimeTrack
//
//  Created by Neil on 11/27/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#ifndef Data_hpp
#define Data_hpp

#include <stdio.h>
#include <armadillo/armadillo>
using namespace arma;

class Data {
public:
    static mat loadVertexs();
    static umat loadFacets();
    static urowvec loadControlPoints();
    static mat loadA();
    static mat loadRealA();
    static mat loadCorner();
    // calculate Rt from uv and mesh vertexes
    static mat loadRt();
    static mat loadRealRt();
};

#endif /* Data_hpp */
