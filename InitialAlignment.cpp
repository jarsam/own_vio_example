//
// Created by liu on 19-12-10.
//

#include "InitialAlignment.h"

void SolveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, std::vector<Eigen::Vector3d> &Bgs)
{
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A.setZero();
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    for(frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); ++frame_i){
        frame_j = next(frame_i);

        Eigen::MatrixXd tmp_A(3, 3);
        Eigen::MatrixXd tmp_b(3);
        tmp_A.setZero();
        tmp_b.setZero();

        Eigen::Quaterniond q_ij(frame_i->second._R.transpose() * frame_j->second._R);
        tmp_A = frame_j->second._pre_integration->_jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second._pre_integration->_delta_q.inverse() * q_ij).vec();
    }
}

bool VisualImuAlignment(std::map<double, ImageFrame> &all_image_frame, std::vector<Eigen::Vector3d> &bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{

}
