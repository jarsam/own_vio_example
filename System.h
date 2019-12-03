#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class System
{
public:
    System(std::string data_path): _data_path(data_path){}

    bool PubImageData();
    bool PubImuData();
    void ProcessBackEnd();
    void Draw();

private:
    void GetImageData(double stamp_sec, cv::Mat &img);
    void GetImuData(double stamp_sec, const Eigen::Vector3d &gyr, const Eigen::Vector3d &acc);

private:
    std::string _data_path;
    const double _delay_times = 2.0;
};