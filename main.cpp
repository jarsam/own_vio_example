#include <GSLAM/core/GSLAM.h>
#include <GSLAM/core/Timer.h>

#include "System.h"

int main(int argc, char **argv)
{
    GSLAM::ScopedTimer("sumTime");
    svar.ParseMain(argc, argv);
    std::string dataPath = svar.GetString("dataset", "");
    std::shared_ptr<System> system(new System(dataPath));

    std::thread thd_BackEnd(&System::ProcessBackEnd, system);
    std::thread thd_PubImuData(&System::PubImuData, system);
    std::thread thd_PubImageData(&System::PubImageData, system);
    std::thread thd_Draw(&System::Draw, system);

    thd_PubImuData.join();
    thd_PubImageData.join();

    std::cout << "main end... see you ..." << std::endl;
    return 0;
}
