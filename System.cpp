#include "System.h"
#include "Parameters.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>
#include <vector>
#include <set>
#include <algorithm>
#include <map>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

void System::ReadParameters()
{
    cv::FileStorage camera_setting(_data_path + "/cam1/sensor.yaml", cv::FileStorage::READ);
    cv::FileStorage imu_setting(_data_path + "/imu0/sensor.yaml", cv::FileStorage::READ);
    if (!camera_setting.isOpened() && !imu_setting.isOpened()){
        std::cerr << "1 ReadParameters Error: wrong path to setting!" << std::endl;
        return;
    }

    std::vector<double> Tic;

    camera_setting["intrinsics"] >> para._camera_intrinsics;
    camera_setting["distortion_coefficients"] >> para._distortion_coefficients;

    para._width = camera_setting["resolution"][0];
    para._height = camera_setting["resolution"][1];
    para._acc_noise = imu_setting["accelerometer_noise_density"];
    para._acc_random = imu_setting["accelerometer_random_walk"];
    para._gyr_noise = imu_setting["gyroscope_noise_density"];
    para._gyr_random = imu_setting["gyroscope_random_walk"];
}

bool System::PubImuData()
{
    std::string sImu_data_file = _data_path + "/imu0/data.csv";
    std::cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << std::endl;
    std::ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open())
    {
        std::cerr << "Failed to open imu file! " << sImu_data_file << std::endl;
        return false;
    }

    std::string sImu_line;
    double dStampNSec = 0.0;
    Eigen::Vector3d vAcc;
    Eigen::Vector3d vGyr;

    std::string firstLine;
    std::getline(fsImu, firstLine);
    while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
    {
        std::vector<double > outputStr;
        std::stringstream ss(sImu_line);
        std::string str;
        while (std::getline(ss, str, ','))
        {
            outputStr.push_back(std::stod(str));
        }
        dStampNSec = outputStr[0];
        vGyr.x() = outputStr[1];
        vGyr.y() = outputStr[2];
        vGyr.z() = outputStr[3];
        vAcc.x() = outputStr[4];
        vAcc.y() = outputStr[5];
        vAcc.z() = outputStr[6];

//        std::cout << "Imu t: " << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << std::endl;
        GetImuData(dStampNSec / 1e9, vGyr, vAcc);
        usleep(5000*_delay_times);
    }
    fsImu.close();
}

bool System::PubImageData()
{
    std::string sImage_file = _data_path + "/cam1/data.csv";
    std::cout << "1 PubImageData start sImage_file: " << sImage_file << std::endl;

    std::ifstream fsImage;
    fsImage.open(sImage_file.c_str());
    if (!fsImage.is_open())
    {
        std::cerr << "Failed to open image file! " << sImage_file << std::endl;
        return false;
    }

    std::string sImage_line;
    double dStampNSec;
    std::string sImgFileName;

    std::string firstLine;
    std::getline(fsImage, firstLine);
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
    {
        std::vector<std::string > outputStr;
        std::stringstream ss(sImage_line);
        std::string str;
        while (std::getline(ss, str, ','))
        {
            outputStr.push_back(str);
        }
        dStampNSec = std::stod(outputStr[0]);
        sImgFileName = outputStr[1];

        std::string imagePath = _data_path + "/cam1/data/" + sImgFileName;
        imagePath.resize(imagePath.size() - 1);
        cv::Mat img = cv::imread(imagePath.c_str(), 0);

        if (img.empty())
        {
            std::cerr << "image is empty! path: " << imagePath << std::endl;
            return false;
        }
        GetImageData(dStampNSec / 1e9, img);
        usleep(50000*_delay_times);
    }

    fsImage.close();
}

void System::GetImuData(double stamp_sec, const Eigen::Vector3d &gyr, const Eigen::Vector3d &acc)
{
    ImuMessagePtr imu_msg(new ImuMessage());
    imu_msg->_header = stamp_sec;
    imu_msg->_linear_acceleration = acc;
    imu_msg->_angular_velocity = gyr;

    if (stamp_sec <= _last_imu_time){
        std::cerr << "imu message in disorder!" << std::endl;
        return;
    }
    _last_imu_time = stamp_sec;

    // FIXME: 感觉这里的Imu的锁应该和特征点的锁不一样.
    _feature_buf_mutex.lock();
    _imu_buf.push(imu_msg);
    _feature_buf_mutex.unlock();
    _con.notify_one();
}

void System::GetImageData(double stamp_sec, cv::Mat &img)
{
    if(!_init_feature) {
        std::cout << "1 GetImageData first detected feature." << std::endl;
        _init_feature = 1;
        return;
    }
    if(_first_image_flag) {
        std::cout << "2 GetImageData first image." << std::endl;
        _first_image_flag = false;
        _first_image_time = stamp_sec;
        _last_image_time = stamp_sec;
        return;
    }
    if(stamp_sec - _last_image_time > 1.0 || stamp_sec < _last_image_time){
        std::cerr << "3 GetImageData discontinue! reset the feature tracker!" << std::endl;
        _first_image_flag = true;
        _last_image_time = 0;
        _pub_count = 1;
    }

    _last_image_time = stamp_sec;

    double MaxFREQ = svar.GetDouble("max_frequency", 10);
    double MinFREQ = svar.GetDouble("min_frequency", 0.1);
    // control the frequency
    if(round(_pub_count / (stamp_sec - _first_image_time)) <= MaxFREQ){
        para._pub_this_frame = true;
        // the frequency is too slow, reset the frequency
        // FIXME: 感觉这里给有问题,不应该减去MaxFREQ
        if(abs(_pub_count / (stamp_sec - _first_image_time) - MaxFREQ) < MinFREQ){
            _first_image_time = stamp_sec;
            _pub_count = 0;
        }
    }
    else{
        para._pub_this_frame = false;
    }

    _tracker_data[0].ReadImage(img, stamp_sec);
    for(int i = 0; ; ++i){
        bool completed = false;
        //更新ID.
        completed = _tracker_data[0].UpdataID(i);

        if (!completed)
            break;
    }
    if (para._pub_this_frame){
        _pub_count++;
        ImageMessagePtr feature_points(new ImageMessage());
        feature_points->_header = stamp_sec;
        std::vector<std::set<int> > hash_ids(svar.GetInt("number_of_camera", 1));
        for(int i = 0; i < hash_ids.size(); ++i){
            auto &un_pts = _tracker_data[i]._cur_un_pts;
            auto &cur_pts = _tracker_data[i]._cur_pts;
            auto &ids = _tracker_data[i]._ids;
            auto &pts_velocity = _tracker_data[i]._pts_velocity;
            for(int j = 0; j < ids.size(); j++){
                // 跟踪数量大于1
                if (_tracker_data[i]._track_cnt[j] > 1){
                    int pt_id = ids[j];
                    hash_ids[i].insert(pt_id);
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->_points.emplace_back(Eigen::Vector3d(x, y, z));
                    feature_points->_points_id.emplace_back(pt_id * svar.GetInt("number_of_camera", 1) + i);
                    feature_points->_point_u.emplace_back(cur_pts[j].x);
                    feature_points->_point_v.emplace_back(cur_pts[j].y);
                    feature_points->_point_x_velocity.emplace_back(pts_velocity[j].x);
                    feature_points->_point_y_velocity.emplace_back(pts_velocity[j].y);
                }
            }
            if (!_init_pub){
                std::cout << "4 GetImageData skip the first image!" << std::endl;
                _init_pub = true;
            }
            else{
                _feature_buf_mutex.lock();
                _feature_buf.push(feature_points);
                _feature_buf_mutex.unlock();
                _con.notify_one();
            }
        }
    }

    cv::Mat show_img;
    cv::cvtColor(img, show_img, CV_GRAY2RGB);
    if (svar.GetInt("show_track", 0)){
        for (int i = 0; i < _tracker_data[0]._cur_pts.size(); ++i){
            double len = std::min(1.0, 1.0 * _tracker_data[0]._track_cnt[i] / svar.GetInt("window_size", 20));
            cv::circle(show_img, _tracker_data[0]._cur_pts[i], 2, cv::Scalar(255 * (1 - len), 0, 255 *len), 2);
        }

        cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
        cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
    }
}

std::vector<std::pair<std::vector<ImuMessagePtr>, ImageMessagePtr>> System::GetMeasurements()
{
    std::vector<std::pair<std::vector<ImuMessagePtr>, ImageMessagePtr>> measurements;
    // 直到把_imu_buf或者_feature_buf中的数据全部取出,才会退出while循环.
    while(true){
        // 如果没有imu的数据和feature的数据都返回
        if (_imu_buf.empty() || _feature_buf.empty()){
            return measurements;
        }
        if (_imu_buf.back()->_header < _feature_buf.front()->_header + _estimator._td){
            std::cerr << "wait for imu" << std::endl;
            return measurements;
        }
        if (_imu_buf.front()->_header > _feature_buf.front()->_header + _estimator._td){
            _feature_buf.pop();
            continue;
        }

        ImageMessagePtr img_msg = _feature_buf.front();
        _feature_buf.pop();

        std::vector<ImuMessagePtr> imu_msg;
        // 一帧图像特征点数据,对应多帧Imu数据,把它们进行对应,然后塞入measurements
        // 一帧图像特征点数据,与它和上一帧图像特征点数据之间的时间间隔内所有Imu数据,以及时间戳晚于当前帧图像的第一帧Imu数据.
        while(_imu_buf.front()->_header < img_msg->_header + _estimator._td){
            imu_msg.emplace_back(_imu_buf.front());
            _imu_buf.pop();
        }
        // 时间戳晚于当前帧图像的第一帧Imu数据也是下一帧图像和当前帧之间Imu数据.
        imu_msg.emplace_back(_imu_buf.front());// 时间戳晚于当前帧图像的第一帧Imu数据
        if(imu_msg.empty()){
            std::cerr << "no imu between two frames" << std::endl;
        }
        // 感觉这个会传输很多信息．
        measurements.emplace_back(imu_msg, img_msg);
    }

    return measurements;
}

void System::ProcessBackEnd()
{
    std::cout << "1 ProcessBackEnd start" << std::endl;
    while (_start_backend){
        std::vector<std::pair<std::vector<ImuMessagePtr>, ImageMessagePtr> > measurements;
        std::unique_lock<std::mutex> lk(_feature_buf_mutex);
        _con.wait(lk, [&]{
            return (measurements = GetMeasurements()).size() != 0;
        });
        if (measurements.size() > 0){
            std::cout << "GetMeasurement size: " << measurements.size();
        }
        lk.unlock();
        _estimator_mutex.lock();
        for(auto &measurement: measurements){
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz =0, rx = 0, ry = 0, rz = 0;
            for(auto &imu_msg: measurement.first){
                double t = imu_msg->_header;
                double img_t = imu_msg->_header + _estimator._td;
                if (t <= img_t){
                    if (_imu_current_time < 0)
                        _imu_current_time = t;
                    double dt = t - _imu_current_time;
                    _imu_current_time = t;
                    dx = imu_msg->_linear_acceleration.x();
                    dy = imu_msg->_linear_acceleration.y();
                    dz = imu_msg->_linear_acceleration.z();
                    rx = imu_msg->_angular_velocity.x();
                    ry = imu_msg->_angular_velocity.y();
                    rz = imu_msg->_angular_velocity.z();
                    _estimator.ProcessIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                }
                else{
                    double dt_1 = img_t - _imu_current_time;
                    double dt_2 = t - img_t;
                    _imu_current_time = img_t;
                    // FIXME: 感觉w1和w2反了
                    double w1 = dt_2 / (dt_2 + dt_1);
                    double w2 = dt_1 / (dt_2 + dt_1);
                    dx = w1 * dx + w2 * imu_msg->_linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->_linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->_linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->_angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->_angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->_angular_velocity.z();
                    // FIXME:感觉应该是dt_1+dt_2.
                    _estimator.ProcessIMU(dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                }
            }
            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for(int i = 0; i < img_msg->_points.size(); ++i){
                // 从1开始
                int id = img_msg->_points_id[i] + 0.5;
                int feature_id = id / svar.GetInt("number_of_camera", 1);
                int camera_id = id % svar.GetInt("number_of_camera", 1);
                double x = img_msg->_points[i].x();
                double y = img_msg->_points[i].y();
                double z = img_msg->_points[i].z();
                double p_u = img_msg->_point_u[i];
                double p_v = img_msg->_point_v[i];
                double velocity_x = img_msg->_point_x_velocity[i];
                double velocity_y = img_msg->_point_y_velocity[i];
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            _estimator.ProcessImage(image, img_msg->_header);
            if(_estimator._solver_flag == Estimator::SolverFlag::NON_LINEAR){

            }
        }
        _estimator_mutex.unlock();
    }
}

void System::Draw()
{
    usleep(5000);
}
