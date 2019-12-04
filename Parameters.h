//
// Created by liu on 19-12-4.
//

#ifndef VIO_EXAMPLE_PARAMETERS_H
#define VIO_EXAMPLE_PARAMETERS_H

#include <vector>

class GlobalParameters
{
private:
    GlobalParameters(){}

public:
    static GlobalParameters& GetInstance(){
        static GlobalParameters instance;
        return instance;
    }

public:
    double _width;
    double _height;
    std::vector<double > _camera_intrinsics;
    std::vector<double > _distortion_coefficients;

public:
    bool _pub_this_frame;
};

#define para GlobalParameters::GetInstance()

#endif //VIO_EXAMPLE_PARAMETERS_H
