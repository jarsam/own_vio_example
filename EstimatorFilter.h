//
// Created by liu on 20-1-15.
//

#ifndef VIO_EXAMPLE_ESTIMATORFILTER_H
#define VIO_EXAMPLE_ESTIMATORFILTER_H

#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

#include <map>
#include <vector>

#include "Estimator.h"

class EstimatorFilter: public Estimator
{
public:
    EstimatorFilter(){}
    ~EstimatorFilter(){}

    void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);
    void ClearState();
};


#endif //VIO_EXAMPLE_ESTIMATORFILTER_H
