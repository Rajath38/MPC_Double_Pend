#include <lcm/lcm-cpp.hpp>
#include "msg/exlcm/example_t.hpp"
#include <chrono>
#include <thread>
#include <iostream>

int main()
{
    lcm::LCM lcm;
    if (!lcm.good()) {
        std::cerr << "LCM initialization failed" << std::endl;
        return 1;
    }

    int32_t counter = 0;

    while (true) {
        exlcm::example_t msg;
        msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count();
        msg.position = counter++;
        msg.name = "example_message";

        lcm.publish("EXAMPLE_CHANNEL", &msg);

        std::cout << "Published message: " << msg.position << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
