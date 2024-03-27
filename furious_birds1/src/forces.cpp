#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"

void processGravityForce(Eigen::VectorXd &F)
{
}

void computeForceAndHessian(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    F.resize(q.size());
    F.setZero();
    H.resize(q.size(), q.size());
    H.setZero();

    std::vector<Eigen::Triplet<double>> Hcoeffs;
    if (params_.gravityEnabled)
        processGravityForce(F);

    H.setFromTriplets(Hcoeffs.begin(), Hcoeffs.end());
}