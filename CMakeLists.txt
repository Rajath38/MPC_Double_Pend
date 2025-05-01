cmake_minimum_required(VERSION 3.10)
project(MPC_double_pendulum)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(LCM_DIR "/usr/local/lib/lcm/cmake/")

<<<<<<< HEAD
find_package(Pinocchio REQUIRED)


=======
>>>>>>> check
# Include folders
include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${LCM_INCLUDE_DIRS}
)

# Source files
set(PUBLISHER_SRC src/publisher.cpp)
set(SUBSCRIBER_SRC src/subscriber.cpp)
<<<<<<< HEAD
set(MODEL src/Dynamic_model/double_pend_cart.cpp)
=======
>>>>>>> check

# Create publisher executable
add_executable(publisher src/publisher.cpp)
target_link_libraries(publisher lcm)

# Create subscriber executable
add_executable(subscriber src/subscriber.cpp)
target_link_libraries(subscriber lcm)

<<<<<<< HEAD
# Create 
add_executable(double_pend_cart src/Dynamic_model/double_pend_cart.cpp)
target_link_libraries(your_target Pinocchio::pinocchio)

=======
>>>>>>> check
