// Copyright 2023 Inria
// SPDX-License-Identifier: BSD-2-Clause

/*
 * In this short script, we show how to compute inverse dynamics (RNEA), i.e.
 * the vector of joint torques corresponding to a given motion.
 */

 #include <iostream>

 #include "pinocchio/algorithm/joint-configuration.hpp"
 #include "pinocchio/algorithm/rnea.hpp"
 #include "pinocchio/parsers/urdf.hpp"
 #include "pinocchio/algorithm/aba.hpp"
 
 int main(int argc, char ** argv)
 {
   using namespace pinocchio;
 
   // Change to your own URDF file here, or give a path as command-line argument
   const std::string urdf_filename = "model/double_pend.urdf";
 
   // Load the URDF model
   Model model;
   pinocchio::urdf::buildModel(urdf_filename, model);
 
   // Build a data frame associated with the model
   Data data(model);


   float time_step = 0.01 ;
   int nsteps = 500 ;

   //initialize the positiona and velocities

   Eigen::Vector3d q0 {0.0, 0.0, 0.5} ; // Initial joint positions (e.g., slider, hinge1, hinge2)
   Eigen::Vector3d v0 {0.0, 0.0, 0.0} ; //  # Initial joint velocities
   Eigen::Vector3d tau0 {0.0, 0.0, 0.0} ; //  # Initial joint velocities


   
   Eigen::Vector3d a1 = pinocchio::aba(model, data, q0, v0, tau0);

    
   // Print out to the vector of joint torques (in N.m)
   std::cout << "Joint torques: " << a1 << std::endl;



   return 0;
 }