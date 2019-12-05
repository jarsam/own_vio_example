//
// Created by liu on 19-12-5.
//

#ifndef VIO_EXAMPLE_ESTIMATOR_H
#define VIO_EXAMPLE_ESTIMATOR_H

#include <Eigen/Dense>

#include <map>
#include <vector>

class Estimator
{
public:
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    Estimator(){}
    void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);

public:
    double _td;

    SolverFlag _solver_flag;
};


#endif //VIO_EXAMPLE_ESTIMATOR_H
