//
// Created by liu on 19-12-11.
//

#include "InitialSfM.h"

bool InitialSfM::Construct(int frame_num, std::vector<Eigen::Quaterniond> &q, std::vector<Eigen::Vector3d> &T, int l,
                           const Eigen::Matrix3d &relative_R, const Eigen::Vector3d relative_T,
                           std::vector<SfMFeature> &sfm_feature, std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
    _feature_num = sfm_feature.size();
    // 共视关系较强的帧
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    // 当前帧的RT
    q[frame_num - 1] = q[l] * Eigen::Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;

    // cam_rotation等代表的是世界坐标系的位姿
    std::vector<Eigen::Matrix3d> cam_rotation(frame_num);
    std::vector<Eigen::Vector3d> cam_translation(frame_num);
    std::vector<Eigen::Quaterniond> cam_quat(frame_num);
    double cam_rotation_ba[frame_num][4];
    double cam_translation_ba[frame_num][3];
    std::vector<Eigen::Matrix<double, 3, 4> > pose(frame_num);

    cam_quat[l] = q[l].inverse();
    cam_rotation[l] = cam_quat[l].toRotationMatrix();
    cam_translation[l] = -1 * (cam_rotation[l] * T[l]);
    pose[l].block<3, 3>(0, 0) = cam_rotation[l];
    pose[l].block<3, 1>(0, 3) = cam_translation[l];

    cam_quat[frame_num - 1] = q[frame_num - 1].inverse();
    cam_rotation[frame_num - 1] = cam_quat[frame_num - 1].toRotationMatrix();
    cam_translation[frame_num - 1] = -1 * (cam_rotation[frame_num - 1] * T[frame_num - 1]);
    pose[frame_num - 1].block<3, 3>(0, 0) = cam_rotation[frame_num - 1];
    pose[frame_num - 1].block<3, 1>(0, 3) = cam_translation[frame_num - 1];

    // 对l帧之后的帧(包括l帧)进行初始化,计算出帧位姿,点云坐标.
    for(int i = l; i < frame_num - 1; ++i){
        if (i > l){
            Eigen::Matrix3d initial_r = cam_rotation[i - 1];
            Eigen::Vector3d initial_t = cam_translation[i - 1];
            if(!SolveFrameByPnP(initial_r, initial_t, i, sfm_feature))
                return false;
            cam_rotation[i] = initial_r;
            cam_translation[i] = initial_t;
            cam_quat[i] = cam_rotation[i];
            pose[i].block<3, 3>(0, 0) = cam_rotation[i];
            pose[i].block<3, 1>(0, 3) = cam_translation[i];
        }

        // FIXME: 不清楚为什么一定要和最后一帧初始化,而不是一帧帧的初始化.
        TriangulateTwoFrames(i, pose[i], frame_num - 1, pose[frame_num - 1], sfm_feature);
    }

    // 遍历(l+1)帧到(frame_num-2)帧,寻找到第l帧的匹配,三角化更多的地图点.
    for(int i = l + 1; i < frame_num - 1; ++i)
        TriangulateTwoFrames(l, pose[l], i, pose[i], sfm_feature);

    // 遍历第l-1帧到第0帧,先通过pnp计算第i帧的位姿,再三角化一些特征点.
    for(int i = l - 1; i >= 0; --i){
        Eigen::Matrix3d initial_r = cam_rotation[i + 1];
        Eigen::Vector3d initial_t = cam_translation[i + 1];
        // FIXME: 如果i!=0的话, 这个地方很难满足, 一旦有一帧不满足就直接挂了, 感觉有问题.
        if (!SolveFrameByPnP(initial_r, initial_t, i, sfm_feature))
            return false;
        cam_rotation[i] = initial_r;
        cam_translation[i] = initial_t;
        cam_quat[i] = cam_rotation[i];
        pose[i].block<3, 3>(0, 0) = cam_rotation[i];
        pose[i].block<3, 1>(0, 3) = cam_translation[i];
        TriangulateTwoFrames(i, pose[i], l, pose[l], sfm_feature);
    }

    // 对sfm_feature中没有被三角化的点,进行三角化
    for(int i = 0; i < _feature_num; ++i){
        if(sfm_feature[i]._state == true)
            continue;
        if(sfm_feature[i]._observation.size() >= 2){
            Eigen::Vector2d point0, point1;
            // 使用最开始观测到的帧和最后观测到的帧进行三角化让视差最大.
            int frame0 = sfm_feature[i]._observation[0].first;
            point0 = sfm_feature[i]._observation[0].second;
            int frame1 = sfm_feature[i]._observation.back().first;
            point1 = sfm_feature[i]._observation.back().second;
            Eigen::Vector3d point_3d;
            TriangulatePoint(pose[frame0], pose[frame1], point0, point1, point_3d);
            sfm_feature[i]._state = true;
            sfm_feature[i]._position[0] = point_3d(0);
            sfm_feature[i]._position[1] = point_3d(1);
            sfm_feature[i]._position[2] = point_3d(2);
        }
    }

    // 对滑动窗口中的所有帧的位姿和3D特征点进行BA优化
    // 优化的是所有帧的位姿和点云位置.
    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
    for(int i = 0; i < frame_num; ++i){
        cam_translation_ba[i][0] = cam_translation[i].x();
        cam_translation_ba[i][1] = cam_translation[i].y();
        cam_translation_ba[i][2] = cam_translation[i].z();
        cam_rotation_ba[i][0] = cam_quat[i].w();
        cam_rotation_ba[i][1] = cam_quat[i].x();
        cam_rotation_ba[i][2] = cam_quat[i].y();
        cam_rotation_ba[i][3] = cam_quat[i].z();
        problem.AddParameterBlock(cam_rotation_ba[i], 4, local_parameterization);
        problem.AddParameterBlock(cam_translation_ba[i], 3);

        if (i == l)
            problem.SetParameterBlockConstant(cam_rotation_ba[i]);
        // 第l帧和第frame_num-1帧的位移不能变化,不然其他帧的位姿和点云位置的尺度就变了.
        if (i == l || i == frame_num - 1)
            problem.SetParameterBlockConstant(cam_translation_ba[i]);
    }
    for(int i = 0; i < _feature_num; ++i){
        if(sfm_feature[i]._state != true)
            continue;
        for(int j = 0; j < sfm_feature[i]._observation.size(); ++j){
            int l = sfm_feature[i]._observation[j].first;
            ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                sfm_feature[i]._observation[j].second.x(),
                sfm_feature[i]._observation[j].second.y());
            problem.AddResidualBlock(cost_function, NULL, cam_rotation_ba[l], cam_translation_ba[l],
                sfm_feature[i]._position);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport();

    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        //cout << "vision only BA converge" << endl;
    }
    else
    {
        std::cout << "vision only BA not converge " << std::endl;
        return false;
    }
    // 从世界坐标系转为局部坐标系了.
    // 之前是第l帧到当前帧的旋转和位移,现在这是当前帧到第l帧的旋转和位移.
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = cam_rotation_ba[i][0];
        q[i].x() = cam_rotation_ba[i][1];
        q[i].y() = cam_rotation_ba[i][2];
        q[i].z() = cam_rotation_ba[i][3];
        q[i] = q[i].inverse();
        //cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
    }
    for (int i = 0; i < frame_num; i++)
    {

        T[i] = -1 * (q[i] * Eigen::Vector3d(cam_translation_ba[i][0], cam_translation_ba[i][1], cam_translation_ba[i][2]));
        //cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
    }
    for (int i = 0; i < (int)sfm_feature.size(); i++)
    {
        if(sfm_feature[i]._state)
            sfm_tracked_points[sfm_feature[i]._id] = Eigen::Vector3d(sfm_feature[i]._position[0], sfm_feature[i]._position[1], sfm_feature[i]._position[2]);
    }
    return true;
}

bool InitialSfM::SolveFrameByPnP(Eigen::Matrix3d &initial_r, Eigen::Vector3d &initial_t, int i,
                                 std::vector<SfMFeature> &sfm_feature)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for(int j = 0; j < _feature_num; ++j){
        if(sfm_feature[j]._state != true)
            continue;
        Eigen::Vector2d point2d;
        for(int k = 0; k < sfm_feature[j]._observation.size(); ++k){
            if(sfm_feature[j]._observation[k].first == i){
                Eigen::Vector2d img_pts = sfm_feature[j]._observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.emplace_back(pts_2);
                cv::Point3f pts_3(sfm_feature[j]._position[0], sfm_feature[j]._position[1], sfm_feature[j]._position[2]);
                pts_3_vector.emplace_back(pts_3);
                break;
            }
        }
    }
    if(pts_2_vector.size() < 15){
        LOG(ERROR) << "unstable features tracking, please slowly move your device";
        if(pts_2_vector.size() < 10)
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(initial_r, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(initial_t, t);
    // 因为2D点使用了归一化平面上的坐标点,所以相机内参矩阵设置为单位阵.
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_success;
    // 使用上一帧的R和T可以加速pnp求解.
    pnp_success = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if (!pnp_success)
        return false;
    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    initial_r = R_pnp;
    initial_t = T_pnp;
    return true;
}

// 这个算出来的也是一个归一化坐标.
void InitialSfM::TriangulatePoint(Eigen::Matrix<double, 3, 4> &pose0, Eigen::Matrix<double, 3, 4> &pose1,
                                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = point0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = point1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = point1[1] * pose1.row(2) - pose1.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

// 三角化点云坐标
void InitialSfM::TriangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &pose0, int frame1,
                                      Eigen::Matrix<double, 3, 4> &pose1, std::vector<SfMFeature> &sfm_feature)
{
    for(int i = 0; i < _feature_num; ++i){
        if (sfm_feature[i]._state == true)
            continue;
        bool has0 = false, has1 = false;
        Eigen::Vector2d point0;
        Eigen::Vector2d point1;
        for(int j = 0; j < sfm_feature[i]._observation.size(); ++j){
            if(sfm_feature[i]._observation[j].first == frame0){
                point0 = sfm_feature[i]._observation[j].second;
                has0 = true;
            }
            if(sfm_feature[i]._observation[j].first == frame1){
                point1 = sfm_feature[i]._observation[j].second;
                has1 = true;
            }
        }
        if (has0 && has1){
            Eigen::Vector3d point_3d;
            // 算出来的是归一化坐标
            TriangulatePoint(pose0, pose1, point0, point1, point_3d);
            sfm_feature[i]._state = true;
            sfm_feature[i]._position[0] = point_3d(0);
            sfm_feature[i]._position[1] = point_3d(1);
            sfm_feature[i]._position[2] = point_3d(2);
        }
    }
}
