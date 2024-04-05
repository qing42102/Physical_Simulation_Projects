#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"

void processGravityForce(const Eigen::VectorXd &trans_pos, Eigen::VectorXd &F)
{
    for (uint i = 0; i < bodies_.size(); i++)
    {
        double mass = bodies_[i]->getTemplate().getVolume() * bodies_[i]->density;
        F(3 * i + 1) += params_.gravityG * mass;
    }
}

void computeForce(const Eigen::VectorXd &trans_pos,
                  const Eigen::VectorXd &angle,
                  Eigen::VectorXd &F_trans,
                  Eigen::VectorXd &F_angle)
{
    F_trans.resize(trans_pos.size());
    F_trans.setZero();
    F_angle.resize(trans_pos.size());
    F_angle.setZero();

    if (params_.gravityEnabled)
        processGravityForce(trans_pos, F_trans);
}