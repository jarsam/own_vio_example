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

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

void System::ReadParameters()
{
    LOG(ERROR) << _data_path + "/cam0/sensor.yaml" << std::endl;

    cv::FileStorage camera_setting(_data_path + "/cam0/sensor.yaml", cv::FileStorage::READ);
    cv::FileStorage imu_setting(_data_path + "/imu0/sensor.yaml", cv::FileStorage::READ);
    if (!camera_setting.isOpened() && !imu_setting.isOpened()){
        std::cerr << "1 ReadParameters Error: wrong path to setting!" << std::endl;
        return;
    }

    camera_setting["intrinsics"] >> para._camera_intrinsics;
    camera_setting["distortion_coefficients"] >> para._distortion_coefficients;
    para._width = camera_setting["resolution"][0];
    para._height = camera_setting["resolution"][1];
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
    std::string sImage_file = _data_path + "/cam0/data.csv";

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

        std::string imagePath = _data_path + "/cam0/data/" + sImgFileName;
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
}
