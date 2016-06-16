//
//  Configuration.hpp
//  RealTimeTrack
//
//  Created by Neil on 5/22/16.
//  Copyright © 2016 Neil. All rights reserved.
//

#ifndef Configuration_hpp
#define Configuration_hpp

#include <stdio.h>
class Configuration {
    
public:
    enum MODE {
        Default, // quick tracking
        Extend // allow same object tracking, but hitting performance
    };
    
    static bool enableDeformTracking;
    static MODE mode;
};

#endif /* Configuration_hpp */
