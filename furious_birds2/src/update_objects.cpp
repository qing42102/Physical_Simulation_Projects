#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"

void buildConfiguration(Eigen::VectorXd &trans_pos,
                        Eigen::VectorXd &trans_vel,
                        Eigen::VectorXd &angle,
                        Eigen::VectorXd &angle_vel)
{
    trans_pos.resize(3 * bodies_.size());
    trans_vel.resize(3 * bodies_.size());
    angle.resize(3 * bodies_.size());
    angle_vel.resize(3 * bodies_.size());

    for (uint i = 0; i < bodies_.size(); i++)
    {
        trans_pos.segment(3 * i, 3) = bodies_[i]->c;
        trans_vel.segment(3 * i, 3) = bodies_[i]->cvel;
        angle.segment(3 * i, 3) = bodies_[i]->theta;
        angle_vel.segment(3 * i, 3) = bodies_[i]->w;
    }
}

void unbuildConfiguration(const Eigen::VectorXd &trans_pos,
                          const Eigen::VectorXd &trans_vel,
                          const Eigen::VectorXd &angle,
                          const Eigen::VectorXd &angle_vel)
{
    for (uint i = 0; i < bodies_.size(); i++)
    {
        bodies_[i]->c = trans_pos.segment(3 * i, 3);
        bodies_[i]->cvel = trans_vel.segment(3 * i, 3);
        bodies_[i]->theta = angle.segment(3 * i, 3);
        bodies_[i]->w = angle_vel.segment(3 * i, 3);
    }
}