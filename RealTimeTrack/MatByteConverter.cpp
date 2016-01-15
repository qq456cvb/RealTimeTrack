//
//  MatByteConverter.cpp
//  RealTimeTrack
//
//  Created by Neil on 12/4/15.
//  Copyright © 2015 Neil. All rights reserved.
//

#include "MatByteConverter.hpp"

Mat MatByteConverter::bytesToMat(byte *bytes, int width, int height, int type)
{
    Mat image = Mat(height, width, CV_8UC4, bytes);
    return image;
}