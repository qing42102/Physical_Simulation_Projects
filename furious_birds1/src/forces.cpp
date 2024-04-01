#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"

/*
    Compute the force of the gravitational potential between two objects.
    The force is
    \frac{G m_1 m_2}{\|\mathbf{c}_1 - \mathbf{c}_2\|^3} \mathbf{c}_1 - \mathbf{c}_2
    where G is the gravitational constant, m_1 and m_2 are the masses of the two objects,
    and \mathbf{c}_1 and \mathbf{c}_2 are the positions of the two objects
*/
void processGravityForce(const Eigen::VectorXd &trans_pos, Eigen::VectorXd &F)
{
    // Pairwise forces between bodies
    for (uint i = 0; i < bodies_.size(); i++)
    {
        for (uint j = i + 1; j < bodies_.size(); j++)
        {
            Eigen::Vector3d c1 = trans_pos.segment<3>(3 * i);
            Eigen::Vector3d c2 = trans_pos.segment<3>(3 * j);
            double dist = (c1 - c2).norm();

            double m1 = bodies_[i]->getTemplate().getVolume() * bodies_[i]->density;
            double m2 = bodies_[j]->getTemplate().getVolume() * bodies_[j]->density;

            Eigen::Vector3d local_force = params_.gravityG * m1 * m2 / pow(dist, 3) * (c1 - c2);

            F.segment<3>(3 * i) += local_force;
            F.segment<3>(3 * j) -= local_force;
        }
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