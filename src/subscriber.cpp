#include <lcm/lcm-cpp.hpp>
#include "msg/exlcm/example_t.hpp"
#include <iostream>

class Handler
{
public:
    void handleMessage(const lcm::ReceiveBuffer* rbuf,
                       const std::string& chan,
                       const exlcm::example_t* msg)
    {
        std::cout << "Received message on channel \"" << chan << "\"" << std::endl;
        std::cout << "Timestamp: " << msg->timestamp << std::endl;
        std::cout << "Position: " << msg->position << std::endl;
        std::cout << "Name: " << msg->name << std::endl;
    }
};

int main()
{
    lcm::LCM lcm;
    if (!lcm.good()) {
        std::cerr << "LCM initialization failed" << std::endl;
        return 1;
    }

    Handler handlerObject;
    lcm.subscribe("EXAMPLE_CHANNEL", &Handler::handleMessage, &handlerObject);

    while (true) {
        lcm.handle();
    }

    return 0;
}
