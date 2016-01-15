//
//  MatByteConverter.hpp
//  RealTimeTrack
//
//  Created by Neil on 12/4/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#ifndef MatByteConverter_hpp
#define MatByteConverter_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
typedef unsigned char byte;

class MatByteConverter
{
public:
    MatByteConverter();
    ~MatByteConverter();
    
    static Mat bytesToMat(byte *bytes, int width, int height, int type = CV_8UC4);
};

#endif /* MatByteConverter_hpp */
