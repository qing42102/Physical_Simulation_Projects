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
        double mass = bodies_[i]->getTemplate().getVolume() * bodies_[i]->density;

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
Eigen::Vector3d newton_method(std::function<Eigen::Vector3d(Eigen::Vector3d)> func,
                              std::function<Eigen::Matrix3d(Eigen::Vector3d)> deriv_func,
                              const Eigen::Vector3d &initial_guess)
{
    Eigen::Vector3d x = initial_guess;

    for (int i = 0; i < params_.NewtonMaxIters; i++)
    {
        Eigen::Vector3d f_val = func(x);

        // Check for convergence
        if (f_val.norm() < params_.NewtonTolerance)
        {
            std::cout << "Newton's method converged in " << i << " iterations\n";
            break;
        }

        Eigen::Matrix3d deriv_val = deriv_func(x);

        // Solve [df] (x_{i+1} - x_i) = -f(x_i)
        Eigen::ColPivHouseholderQR<Eigen::Matrix3d> solver(deriv_val);
        Eigen::Vector3d diff_x = solver.solve(-f_val);
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
                  const Eigen::VectorXd &angle_vel)
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

/*
    Compute the update for the angular velocity using the formula
    -\rho (\omega^{i+1})^T \mathbf{M}_I \mathbf{T}(-h \omega^{i+1})^{-1}
        + \rho (\omega^i)^T \mathbf{M}_I \mathbf{T}(h \omega^i)^{-1}
        - h [d_{\theta} V(\mathbf{c}^{i+1}, \theta^{i+1})] \mathbf{T}(\theta^{i+1})^{-1}

    Solves for the angular velocity using Newton's method
*/
void update_angle_vel(const Eigen::VectorXd &angle,
                      Eigen::VectorXd &angle_vel,
                      const Eigen::VectorXd &F)
{
    for (uint i = 0; i < bodies_.size(); i++)
    {
        Eigen::Vector3d angle_i = angle.segment(3 * i, 3);
        Eigen::Vector3d angle_vel_i = angle_vel.segment(3 * i, 3);

        Eigen::Matrix3d inertia_tensor = bodies_[i]->density * bodies_[i]->getTemplate().getInertiaTensor();
        Eigen::Matrix3d T1 = VectorMath::TMatrix(params_.timeStep * angle_vel_i);
        Eigen::Matrix3d T2 = VectorMath::TMatrix(angle_i);

        Eigen::Vector3d term1 = angle_vel_i.transpose() * inertia_tensor * T1.inverse();
        Eigen::Vector3d term2 = params_.timeStep * F.segment(3 * i, 3).transpose() * T2.inverse();

        // Lambda function for the angular velocity update and its derivative
        auto func = [&](Eigen::Vector3d new_angle_vel) -> Eigen::Vector3d
        {
            Eigen::Matrix3d T3 = VectorMath::TMatrix(-params_.timeStep * new_angle_vel);
            Eigen::Vector3d term3 = new_angle_vel.transpose() * inertia_tensor * T3.inverse();

            return -term3 + term1 - term2;
        };

        // Note that the second derivative term is small and can be ignored
        auto deriv = [&](Eigen::Vector3d new_angle_vel) -> Eigen::Matrix3d
        {
            Eigen::Matrix3d T3 = VectorMath::TMatrix(-params_.timeStep * new_angle_vel);

            return -inertia_tensor * T3.inverse();
        };

        Eigen::Vector3d initial_state = angle_vel_i;
        Eigen::Vector3d state = newton_method(func, deriv, initial_state);

        angle_vel.segment(3 * i, 3) = state;
    }
}

void numericalIntegration(Eigen::VectorXd &trans_pos,
                          Eigen::VectorXd &trans_vel,
                          Eigen::VectorXd &angle,
                          Eigen::VectorXd &angle_vel)
{
    trans_pos = trans_pos + params_.timeStep * trans_vel;

    update_angle(angle, angle_vel);

    Eigen::VectorXd F_trans, F_angle;
    computeForce(trans_pos, angle, F_trans, F_angle);

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Minv;
    computeMassInverse(Minv);
    trans_vel = trans_vel - Minv * params_.timeStep * F_trans;

    update_angle_vel(angle, angle_vel, F_angle);

    std::cout << "Numerically integrate to update the position and velocity\n";
}