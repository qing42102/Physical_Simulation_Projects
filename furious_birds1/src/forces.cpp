#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"

void processGravityForce(Eigen::VectorXd &F)
{
}

void computeForceAndHessian(const Eigen::VectorXd &trans_pos,
                            const Eigen::VectorXd &trans_vel,
                            const Eigen::VectorXd &angle,
                            const Eigen::VectorXd &angle_vel,
                            Eigen::VectorXd &F,
                            Eigen::SparseMatrix<double> &H)
{
    F.resize(trans_pos.size());
    F.setZero();
    H.resize(trans_pos.size(), trans_pos.size());
    H.setZero();

    std::vector<Eigen::Triplet<double>> Hcoeffs;
    if (params_.gravityEnabled)
        processGravityForce(F);

    H.setFromTriplets(Hcoeffs.begin(), Hcoeffs.end());
}