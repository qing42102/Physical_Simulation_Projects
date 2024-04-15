#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"

void processGravityForce(const Eigen::MatrixXd &Q,
                         Eigen::MatrixXd &force)
{
    for (uint i = 0; i < Q.rows(); i++)
    {
        force(i, 1) += params_.gravityG;
    }
}

void computeForce(const Eigen::MatrixXd &Q,
                  Eigen::MatrixXd &force)
{
    force.resize(Q.rows(), 3);
    force.setZero();

    if (params_.gravityEnabled)
        processGravityForce(Q, force);
}