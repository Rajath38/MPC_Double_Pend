cmake_minimum_required(VERSION 3.10)
project(MPC_double_pendulum)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(LCM_DIR "/usr/local/lib/lcm/cmake/")
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)



# Include folders
include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${LCM_INCLUDE_DIRS}
    ${PINOCCHIO_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(PUBLISHER_SRC src/publisher.cpp)
set(SUBSCRIBER_SRC src/subscriber.cpp)

# Create publisher executable
add_executable(publisher src/publisher.cpp)
target_link_libraries(publisher lcm)

# Create subscriber executable
add_executable(subscriber src/subscriber.cpp)
target_link_libraries(subscriber lcm)

link_directories(/opt/openrobots/lib)

add_executable(double_pend_cart src/Dynamic_model/double_pend_cart.cpp)
target_link_libraries(double_pend_cart pinocchio_default pinocchio_parsers pinocchio_casadi pinocchio_collision)


