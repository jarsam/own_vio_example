#include "System.h"

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

        // cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
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

}
