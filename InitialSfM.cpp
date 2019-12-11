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
        }

        // FIXME: 不清楚为什么一定要和最后一帧初始化,而不是一帧帧的初始化.
        TriangulateTwoFrames(i, pose[i], frame_num - 1, pose[frame_num - 1], sfm_feature);
    }
}

// 通过PnP的方法
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
    
}

// 这个算出来的也是一个归一化坐标.
void InitialSfM::TriangulatePoint(Eigen::Matrix<double, 3, 4> &pose0, Eigen::Matrix<double, 3, 4> &pose1,
                                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = point0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = point1[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(3) = point1[1] * pose0.row(2) - pose0.row(1);

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
