//
// Created by liu on 19-12-6.
//

#ifndef VIO_EXAMPLE_FEATUREMANAGER_H
#define VIO_EXAMPLE_FEATUREMANAGER_H

#include "Parameters.h"

#include <Eigen/Dense>

class FeaturePerFrame
{
public:

};

class FeaturePerId
{

};

class FeatureManager
{
public:
    FeatureManager(std::vector<Eigen::Matrix3d> &Rs);
};


#endif //VIO_EXAMPLE_FEATUREMANAGER_H
