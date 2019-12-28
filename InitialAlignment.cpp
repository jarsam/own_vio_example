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
        Eigen::VectorXd tmp_b(3);
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
    // _Bgs的值本来就是0.
    // FIXME: 但是如果后面初始化失败了, _Bgs不是错误赋值了吗?
    for(int i = 0; i <= svar.GetInt("window_size", 20); ++i)
        Bgs[i] += delta_bg;
    // 同时利用新的Bias重新repropagate
    for(frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); ++frame_i){
        frame_j = next(frame_i);
        frame_j->second._pre_integration->Repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
    }
}

// 求解重力向量在切平面上的两个分量.
// FIXME: 需要之后打印这个函数中的数据进行理解.
Eigen::MatrixXd TangentBasis(Eigen::Vector3d &g0){
    Eigen::Vector3d b, c;
    Eigen::Vector3d a = g0.normalized();
    // 指向切平面的向量tmp
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    // 将原本的重力向量投影到切平面上会生成两个正交的向量b, c.
    // 这样重力向量就分成了tmp, b, c三个向量了.
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

// 将重力的模看做是已知的, 这样重力向量的自由度就从3变成2了.
// 通过重力的模精调速度重力向量和尺度因子

// 除了这个方法以外, 还有一个方法, 那就是直接在残差约束中加一个g-9.8
// 但是这样的话, 雅克比矩阵中就有了与自变量有关的元素, 就不是线性的了, 这样的话, 算起来会比较麻烦, 可能会迭代很多次.
void RefineGravity(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    // g0为9.8乘算出的重力向量
    Eigen::Vector3d g0 = g.normalized() * para._G.norm();
    Eigen::Vector3d lx, ly;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    // 优化了四次后认为这些参数优化到了一个很好的值了.
    for(int k = 0; k < 4; ++k){
        Eigen::MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        // 使用投影到切平面的重力向量重新代入公式进行调整.
        for(frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); ++frame_i, ++i){
            frame_j = next(frame_i);
            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();
            double dt = frame_j->second._pre_integration->_sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second._R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second._R.transpose() * (frame_j->second._T - frame_i->second._T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second._pre_integration->_delta_p +
                                      frame_i->second._R.transpose() * frame_j->second._R * para._Tic - para._Tic -
                                      frame_i->second._R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second._R.transpose() * frame_j->second._R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second._R.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second._pre_integration->_delta_v -
                                      frame_i->second._R.transpose() * dt * Eigen::Matrix3d::Identity() * g0;
            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        Eigen::VectorXd dg = x.segment<2>(n_state-3);
        // 合成为第l帧坐标系下的g0
        g0 = (g0 + lxly * dg).normalized() * para._G.norm();
    }
    g = g0;
}

// 初始化速度,重力向量g和尺度因子s
bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    // 状态量的个数[V1, V2, ..., Vn, g, s]
    // V为速度, g为重力向量, s为尺度.
    int n_state = all_frame_count * 3 + 3 + 1;
    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();
    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for(frame_i = all_image_frame.begin(); std::next(frame_i) != all_image_frame.end(); ++frame_i, ++i){
        frame_j = std::next(frame_i);
        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second._pre_integration->_sum_dt;
        // 使用第l帧坐标系下的坐标算出来的值和Imu预积分出来的值理论上是相等的
        // 所以通过求最小二乘解就可得出相机坐标系下的值.
        // 比如: 速度, 尺度和重力向量.
        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second._R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        // 这里除了100, 所以优化后的尺度也要除以100, 这样是为了让尺度的精度更高.
        tmp_A.block<3, 1>(0, 9) = frame_i->second._R.transpose() * (frame_j->second._T - frame_i->second._T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second._pre_integration->_delta_p + frame_i->second._R.transpose() * frame_j->second._R * para._Tic - para._Tic;

        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second._R.transpose() * frame_j->second._R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second._R.transpose() * dt * Eigen::Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second._pre_integration->_delta_v;

        // FIXME: 这里本来不应该为单位阵的,有改进的空间.
        Eigen::Matrix<double, 6, 6> con_inv = Eigen::Matrix<double, 6, 6>::Identity();
        // 原本为Hx=b,进行下一步为H^THx=H^Tb
        Eigen::MatrixXd r_A = tmp_A.transpose() * con_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * con_inv * tmp_b;

        // A的上面为速度
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();
        // A的下面为重力向量和尺度
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();
        // 这个部分是速度和重力向量,尺度相乘的部分.
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    // FIXME: 为什么要除100?
    // 据说是因为把尺度求导得到的雅克比除以100, 这就意味着, 尺度这个变量对残差的影响力减弱了100倍,
    // 最终为了能够消除残差, 优化后的尺度会比实际的大100倍, 所以得到后要再除以100, 这样做的目的是能够让尺度的精度更高.
    double s = x(n_state - 1) / 100.0;
    g = x.segment<3>(n_state - 4);
    // 只有当优化后的重力接近9.8, 才认为这个重力的向量是好的.
    if (fabs(g.norm() - para._G.norm()) > 1.0 || s < 0)
        return false;
    RefineGravity(all_image_frame, g, x);
    // FIXME: 和上面的问题一样
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    return (s >= 0.0);
}

/*
 * 通过之前初始化获得的相机和Imu的旋转和读取的相机和Imu的位移获得所有滑动窗口的位姿.
 *
 * 在纯视觉初始化的时候,采用第一帧C0时的相机坐标系作为参考坐标系,
 * 通过纯视觉SfM可以获得所有滑动窗口中的位姿,
 * 其中位移向量是没有绝对尺度信息的在C0坐标系的坐标,旋转向量为到C0的旋转四元数.
 */
bool VisualImuAlignment(std::map<double, ImageFrame> &all_image_frame, std::vector<Eigen::Vector3d> &bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    // 使用两帧的相机到Imu的旋转来标定Imu的bias
    SolveGyroscopeBias(all_image_frame, bgs);
    // 使用预积分量初始化速度,中立向量g和尺度因子s
    return LinearAlignment(all_image_frame, g, x);
}
