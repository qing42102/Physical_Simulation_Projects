#include <iostream>
#include <functional>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "forces.cpp"
#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"

void computeMassInverse(Eigen::DiagonalMatrix<double, Eigen::Dynamic> &Minv)
{
    Minv.resize(3 * bodies_.size());

    for (uint i = 0; i < bodies_.size(); i++)
    {
        float mass = bodies_[i]->getTemplate().getVolume() * bodies_[i]->density;

        Minv.diagonal().segment(3 * i, 3) = Eigen::Vector3d::Constant(1.0 / mass);
    }
}

/*
    Newton's method for solving a system of nonlinear equations
    @param func the function whose root we want to find
    @param deriv_func the derivative of the function
    @param initial_guess the initial guess for the root
    @returns the root of the function
*/
Eigen::VectorXd newton_method(std::function<Eigen::VectorXd(Eigen::VectorXd)> func,
                              std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd)> deriv_func,
                              const Eigen::VectorXd &initial_guess)
{
    Eigen::VectorXd x = initial_guess;

    for (int i = 0; i < params_.NewtonMaxIters; i++)
    {
        Eigen::VectorXd f_val = func(x);

        // Check for convergence
        if (f_val.norm() < params_.NewtonTolerance)
        {
            std::cout << "Newton's method converged in " << i << " iterations\n";
            break;
        }

        Eigen::SparseMatrix<double> deriv_val = deriv_func(x);

        // Solve [df] (x_{i+1} - x_i) = -f(x_i)
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(deriv_val);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Decomposition failed\n";
            exit(1);
        }
        Eigen::VectorXd diff_x = solver.solve(-f_val);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Solving failed\n";
            exit(1);
        }
        x = x + diff_x;
    }

    return x;
}

/*
    Compute the update for the angle using the formula
    \text{rot} (\theta^i) \text{rot} (h \omega^i) = \text{rot} (\theta^{i+1})
*/
void update_angle(Eigen::VectorXd &angle,
                  Eigen::VectorXd &angle_vel)
{
    for (uint i = 0; i < bodies_.size(); i++)
    {
        Eigen::Vector3d angle_i = angle.segment(3 * i, 3);
        Eigen::Vector3d angle_vel_i = angle_vel.segment(3 * i, 3);

        Eigen::Matrix3d rotation_mat = VectorMath::rotationMatrix(angle_i);
        Eigen::Matrix3d rotation_mat_change = VectorMath::rotationMatrix(params_.timeStep * angle_vel_i);
        Eigen::Vector3d new_angle = VectorMath::axisAngle(rotation_mat * rotation_mat_change);

        angle.segment(3 * i, 3) = new_angle;
    }
}

void numericalIntegration(Eigen::VectorXd &trans_pos,
                          Eigen::VectorXd &trans_vel,
                          Eigen::VectorXd &angle,
                          Eigen::VectorXd &angle_vel)
{
    trans_pos = trans_pos + params_.timeStep * trans_vel;

    update_angle(angle, angle_vel);

    Eigen::VectorXd F;
    Eigen::SparseMatrix<double> H;
    computeForceAndHessian(trans_pos, trans_vel, angle, angle_vel, F, H);

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Minv;
    computeMassInverse(Minv);
    trans_vel = trans_vel - Minv * params_.timeStep * F;
}