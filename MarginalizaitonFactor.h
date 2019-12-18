//
// Created by liu on 19-12-18.
//

#ifndef VIO_EXAMPLE_MARGINALIZAITONFACTOR_H
#define VIO_EXAMPLE_MARGINALIZAITONFACTOR_H

#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "Utility.h"

class MarginalizationInfo
{
public:
};

class MarginalizaitonFactor: public ceres::CostFunction
{
public:
    MarginalizaitonFactor(MarginalizationInfo* marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians)const;

    MarginalizationInfo* _marginalization_info;
};


#endif //VIO_EXAMPLE_MARGINALIZAITONFACTOR_H
