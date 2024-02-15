#include <iostream>
#include <functional>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

SimParameters params_;
std::vector<Particle, Eigen::aligned_allocator<Particle>> particles_;
std::vector<Spring *> connectors_;
std::vector<Saw> saws_;

void computeMassInverse(Eigen::SparseMatrix<double> &Minv)
{
    // Minv is a diagonal matrix with the inverse mass of each particle on the diagonal
    // Its size is 2 * particles_.size() x 2 * particles_.size()

    // Populate Minv with the inverse mass matrix
    Minv.reserve(Eigen::VectorXi::Constant(particles_.size(), 1));
    for (uint i = 0; i < particles_.size(); i++)
    {
        Minv.insert(2 * i, 2 * i) = 1.0 / particles_[i].mass;
        Minv.insert(2 * i + 1, 2 * i + 1) = 1.0 / particles_[i].mass;
    }

    Minv.makeCompressed();
}

/*
    Add the force due to gravity to the force vector F
    Add the Hessian of the gravity force to the Hessian matrix H
    V = \sum_i m_i * g * y_i
    F = [0, \sum_i m_i * g]
    H = [0, 0, 0, 0]
*/
void compute_gravity_force(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    if (params_.gravityEnabled)
    {
        for (uint i = 0; i < particles_.size(); i++)
        {
            if (!particles_[i].fixed)
            {
                F(2 * i) += 0;
                F(2 * i + 1) += -params_.gravityG * particles_[i].mass;
            }
        }
    }
}

/*
    Add the force due to spring to the force vector F
    Add the Hessian of the spring force to the Hessian matrix H
    V = \sum_{ij} \frac{1}{2} k_{ij} (\Vert p_i - p_j \Vert - L_{ij})^2
    F = \sum_{ij} -k_{ij} (\Vert p_i - p_j \Vert - L_{ij}) \frac{p_i - p_j}{\Vert p_i - p_j \Vert}
    H = \sum_{ij} k_{ij} \left(\frac{(p_i - p_j) (p_i - p_j)^T}{\Vert p_i - p_j \Vert^2} + (\Vert p_i - p_j \Vert - L_{ij}) \left(\frac{\mathbf{I}}{\Vert p_i - p_j \Vert} - \frac{(p_i - p_j) (p_i - p_j)^T}{\Vert p_i - p_j \Vert^3} \right) \right)
*/
void compute_spring_force(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    if (params_.springsEnabled)
    {
        for (uint i = 0; i < connectors_.size(); i++)
        {
            double L = connectors_[i]->restlen;
            double k = connectors_[i]->stiffness / L;

            int p1_index = 2 * connectors_[i]->p1;
            int p2_index = 2 * connectors_[i]->p2;

            Eigen::Vector2d pos1 = q.segment(2 * p1_index, 2);
            Eigen::Vector2d pos2 = q.segment(2 * p2_index, 2);

            double dist = (pos1 - pos2).norm();

            Eigen::Vector2d pos_unit = (pos1 - pos2) / dist;

            if (!particles_[p1_index].fixed)
            {
                F.segment(2 * p1_index, 2) += -k * (dist - L) * pos_unit;
                H.insert(2 * connectors_[i]->p1, 2 * connectors_[i]->p1) = k * (1 - (pos1 - pos2).dot(pos1 - pos2) / (dist * dist));
            }
            if (!particles_[p2_index].fixed)
            {
                F.segment(2 * p2_index, 2) += k * (dist - L) * (pos1 - pos2) / dist;
                H.insert(2 * connectors_[i]->p2, 2 * connectors_[i]->p2) = k * (1 - (pos1 - pos2).dot(pos1 - pos2) / (dist * dist));
            }
        }
    }
}

void computeForceAndHessian(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    // Compute the total force and Hessian for all potentials in the system
    // This function should respect the booleans in params_ to allow the user
    // to toggle on and off individual force types.

    // Clear the force vector and Hessian matrix for the new step
    F.setZero();
    H.setZero();

    compute_gravity_force(q, qprev, F, H);
    compute_spring_force(q, qprev, F, H);
}

Eigen::VectorXd newton_method(std::function<Eigen::VectorXd(Eigen::VectorXd)> func,
                              std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd)> deriv_func,
                              const Eigen::VectorXd &initial_guess)
{
    Eigen::VectorXd x = initial_guess;

    for (int i = 0; i < params_.NewtonMaxIters; i++)
    {
        Eigen::VectorXd f_val = func(x);
        if (f_val.norm() < params_.NewtonTolerance)
        {
            std::cout << "Newton's method converged in " << i << " iterations" << std::endl;
            break;
        }

        Eigen::SparseMatrix<double> deriv_val = deriv_func(x);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(deriv_val);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Decomposition failed" << std::endl;
            exit(1);
        }
        x = solver.solve(x - f_val);
    }

    return x;
}

void numericalIntegration(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot)
{
    Eigen::SparseMatrix<double> Minv(2 * particles_.size(), 2 * particles_.size());
    computeMassInverse(Minv);

    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    std::cout << "One step of integrator" << std::endl;

    // Perform one step of time integration, using the method in params_.integrator
    switch (params_.integrator)
    {
    case SimParameters::TI_EXPLICIT_EULER:
    {
        computeForceAndHessian(q, qprev, F, H);
        qprev = q;
        q = q + params_.timeStep * Minv * qdot.transpose();
        qdot = qdot + params_.timeStep * F;
    }

    case SimParameters::TI_IMPLICIT_EULER:
    {
        qprev = q;
        auto implicit_euler = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian(q, qprev, F, H);
            return q - qprev - params_.timeStep * Minv * (qdot.transpose() + params_.timeStep * F.transpose());
        };

        auto implicit_euler_deriv = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian(q, qprev, F, H);
            return Eigen::MatrixXd::Identity(q.size(), q.size()) - pow(params_.timeStep, 2) * Minv * H;
        };

        q = newton_method(implicit_euler, implicit_euler_deriv, q);
    }

    case SimParameters::TI_IMPLICIT_MIDPOINT:
    {
        qprev = q;
        auto implicit_midpoint = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian((q + qprev) / 2, qprev, F, H);
            return q - qprev - params_.timeStep * Minv * (2 * qdot.transpose() + params_.timeStep * F.transpose()) / 2;
        };

        auto implicit_midpoint_deriv = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian((q + qprev) / 2, qprev, F, H);
            return Eigen::MatrixXd::Identity(q.size(), q.size()) - 0.5 * pow(params_.timeStep, 2) * Minv * H;
        };

        q = newton_method(implicit_midpoint, implicit_midpoint_deriv, q);
    }

    case SimParameters::TI_VELOCITY_VERLET:
    {
        qprev = q;
        q = q + params_.timeStep * Minv * qdot.transpose();
        computeForceAndHessian(q, qprev, F, H);
        qdot = qdot + params_.timeStep * F;
    }

    default:
        std::cerr << "Invalid time integrator" << std::endl;
    }
}
