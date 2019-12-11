//
// Created by liu on 19-12-11.
//

#ifndef VIO_EXAMPLE_INITIALSFM_H
#define VIO_EXAMPLE_INITIALSFM_H

#include <iostream>
#include <deque>
#include <map>
#include <cstdlib>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

struct SfMFeature
{
    bool _state;
    int _id;
    std::vector<std::pair<int, Eigen::Vector2d>> _observation;
    double _position[3];
    double _depth;
};

struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v): _observed_u(observed_u), _observed_v(observed_v){}

    template <typename T>
    bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(_observed_u);
        residuals[1] = yp - T(_observed_v);
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<
            ReprojectionError3D, 2, 4, 3, 3>(
            new ReprojectionError3D(observed_x,observed_y)));
    }

    double _observed_u;
    double _observed_v;
};

class InitialSfM
{
public:
    InitialSfM(){}
    bool Construct(int frame_num, std::vector<Eigen::Quaterniond> &q, std::vector<Eigen::Vector3d> &T, int l, const Eigen::Matrix3d &relative_R,
                   const Eigen::Vector3d relative_T, std::vector<SfMFeature> &sfm_feature, std::map<int, Eigen::Vector3d> &sfm_tracked_points);

private:
    bool SolveFrameByPnP(Eigen::Matrix3d &initial_r, Eigen::Vector3d &initial_t, int i, std::vector<SfMFeature> &sfm_feature);
    void TriangulatePoint(Eigen::Matrix<double, 3, 4> &pose0, Eigen::Matrix<double, 3, 4> &pose1,
                              Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void TriangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &pose0, int frame1, Eigen::Matrix<double, 3, 4> &pose1,
                              std::vector<SfMFeature> &sfm_feature);

private:
    int _feature_num;
};


#endif //VIO_EXAMPLE_INITIALSFM_H

