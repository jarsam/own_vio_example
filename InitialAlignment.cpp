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

        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }

    // 这里本质上是一个最小二乘问题,如果没有预积分,则需要用ceres进行优化.
    Eigen::Vector3d delta_bg = A.ldlt().solve(b);
    // 这里求出来的只是Bias的变化量,所以要进行累加.
    for(int i = 0; i <= svar.GetInt("window_size", 10); ++i)
        Bgs[i] += delta_bg;
    // 同时利用新的Bias重新repropagate
    for(frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); ++frame_i){
        frame_j = next(frame_i);
        frame_j->second._pre_integration->Repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
    }
}

// 初始化速度,重力向量g和尺度因子
bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1; // 状态量的个数[V1, V2, ..., Vn, g, s]
    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
}

bool VisualImuAlignment(std::map<double, ImageFrame> &all_image_frame, std::vector<Eigen::Vector3d> &bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    SolveGyroscopeBias(all_image_frame, bgs);
    return LinearAlignment(all_image_frame, g, x);
}
