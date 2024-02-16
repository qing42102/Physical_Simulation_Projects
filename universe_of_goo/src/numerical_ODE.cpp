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
    Minv.reserve(Eigen::VectorXi::Constant(2 * particles_.size(), 1));
    for (uint i = 0; i < particles_.size(); i++)
    {
        Minv.insert(2 * i, 2 * i) = 1.0 / particles_[i].mass;
        Minv.insert(2 * i + 1, 2 * i + 1) = 1.0 / particles_[i].mass;
    }
}

/*
    Add the force due to gravity to the force vector F
    Add the Hessian of the gravity force to the Hessian matrix H
    V = -\sum_i m_i * g * y_i
    F = [0, \sum_i m_i * g]
    H = [0, 0, 0, 0]
*/
void compute_gravity_force(const Eigen::VectorXd &q, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{

    for (uint i = 0; i < particles_.size(); i++)
    {
        if (!particles_[i].fixed)
        {
            F(2 * i) += 0;
            F(2 * i + 1) += params_.gravityG * particles_[i].mass;
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
void compute_spring_force(const Eigen::VectorXd &q, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{

    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(2 * particles_.size());

    for (uint i = 0; i < connectors_.size(); i++)
    {
        double L = connectors_[i]->restlen;
        double k = connectors_[i]->stiffness / L;

        int p1_index = connectors_[i]->p1;
        int p2_index = connectors_[i]->p2;

        Eigen::Vector2d pos1 = q.segment(2 * p1_index, 2);
        Eigen::Vector2d pos2 = q.segment(2 * p2_index, 2);

        double dist = (pos1 - pos2).norm();

        Eigen::Vector2d pos_unit = (pos1 - pos2) / dist;

        // Local Hessian matrix
        Eigen::Matrix2d H_value = pos_unit * pos_unit.transpose() + (dist - L) * (Eigen::MatrixXd::Identity(2, 2) / dist - pos_unit * pos_unit.transpose() / dist);

        // Local force vector
        Eigen::VectorXd F_value = k * (dist - L) * pos_unit;

        if (!particles_[p1_index].fixed)
        {
            // Corresponding index in the force vector
            F.segment(2 * p1_index, 2) += -F_value;

            // Corresponding index in the Hessian matrix
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index, 2 * p1_index, -H_value(0, 0)));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index + 1, 2 * p1_index, -H_value(1, 0)));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index, 2 * p1_index + 1, -H_value(0, 1)));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index + 1, 2 * p1_index + 1, -H_value(1, 1)));
        }
        if (!particles_[p2_index].fixed)
        {
            F.segment(2 * p2_index, 2) += F_value;

            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index, 2 * p2_index, H_value(0, 0)));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index + 1, 2 * p2_index, H_value(1, 0)));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index, 2 * p2_index + 1, H_value(0, 1)));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index + 1, 2 * p2_index + 1, H_value(1, 1)));
        }
    }

    H.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

/*
    Add the normal force due to the floor
*/
void floor_force(Eigen::VectorXd &q, Eigen::VectorXd &qdot, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    for (uint i = 0; i < particles_.size(); i++)
    {
        if (particles_[i].pos(1) <= -0.5 && !particles_[i].fixed)
        {
            F(2 * i) += 0;
            F(2 * i + 1) += params_.gravityG * particles_[i].mass;

            q(2 * i + 1) = -0.5;
            qdot(2 * i + 1) = 0;
        }
    }
}

void computeForceAndHessian(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    // Compute the total force and Hessian for all potentials in the system
    // This function should respect the booleans in params_ to allow the user
    // to toggle on and off individual force types.

    // Clear the force vector and Hessian matrix for the new step
    F.setZero();
    H.setZero();

    if (params_.gravityEnabled)
    {
        compute_gravity_force(q, F, H);
    }
    if (params_.springsEnabled)
    {
        compute_spring_force(q, F, H);
    }
    if (params_.floorEnabled)
    {
        floor_force(q, qdot, F, H);
    }
}

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
            std::cout << "Newton's method converged in " << i << " iterations" << std::endl;
            break;
        }

        Eigen::SparseMatrix<double> deriv_val = deriv_func(x);

        // Solve [df] (x_{i+1} - x_i) = -f(x_i)
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(deriv_val);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Decomposition failed" << std::endl;
            exit(1);
        }
        Eigen::VectorXd diff_x = solver.solve(-f_val);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Solving failed" << std::endl;
            exit(1);
        }
        x = x + diff_x;
    }

    return x;
}

void numericalIntegration(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot)
{
    if (q.size() == 0)
    {
        return;
    }

    Eigen::SparseMatrix<double> Minv(2 * particles_.size(), 2 * particles_.size());
    computeMassInverse(Minv);

    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    // Perform one step of time integration, using the method in params_.integrator
    switch (params_.integrator)
    {
    case SimParameters::TI_EXPLICIT_EULER:
    {
        computeForceAndHessian(q, qprev, qdot, F, H);
        qprev = q;
        q = q + params_.timeStep * qdot;
        qdot = qdot + params_.timeStep * Minv * F;

        std::cout << "One step of Explicit Euler" << std::endl;
        break;
    }

    case SimParameters::TI_IMPLICIT_EULER:
    {
        qprev = q;
        // Lambda function for the implicit Euler method and its derivative
        auto implicit_euler = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian(q, qprev, qdot, F, H);
            return q - qprev - params_.timeStep * (qdot + params_.timeStep * Minv * F);
        };

        auto implicit_euler_deriv = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian(q, qprev, qdot, F, H);
            Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(2 * particles_.size(), 2 * particles_.size()).sparseView();
            Eigen::SparseMatrix<double> deriv = sparse_I - pow(params_.timeStep, 2) * Minv * H;
            return deriv;
        };

        q = newton_method(implicit_euler, implicit_euler_deriv, q);

        std::cout << "One step of Implicit Euler" << std::endl;
        break;
    }

    case SimParameters::TI_IMPLICIT_MIDPOINT:
    {
        qprev = q;
        Eigen::VectorXd avg_q = (q + qprev) / 2;
        // Lambda function for the implicit midpoint method and its derivative
        auto implicit_midpoint = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian(avg_q, qprev, qdot, F, H);
            return q - qprev - params_.timeStep * Minv * (2 * qdot + params_.timeStep * F) / 2;
        };

        auto implicit_midpoint_deriv = [&](Eigen::VectorXd q)
        {
            computeForceAndHessian(avg_q, qprev, qdot, F, H);
            Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(2 * particles_.size(), 2 * particles_.size()).sparseView();
            return sparse_I - 0.5 * pow(params_.timeStep, 2) * Minv * H;
        };

        q = newton_method(implicit_midpoint, implicit_midpoint_deriv, q);

        std::cout << "One step of Implicit Midpoint" << std::endl;
        break;
    }

    case SimParameters::TI_VELOCITY_VERLET:
    {
        qprev = q;
        q = q + params_.timeStep * qdot;
        computeForceAndHessian(q, qprev, qdot, F, H);
        qdot = qdot + params_.timeStep * Minv * F;

        std::cout << "One step of Velocity Verlet" << std::endl;
        break;
    }

    default:
        std::cerr << "Invalid time integrator" << std::endl;
    }
}
