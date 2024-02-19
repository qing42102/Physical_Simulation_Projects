#include <iostream>
#include <functional>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

#include "forces_hessian.cpp"

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

void explicit_euler(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot, Eigen::SparseMatrix<double> &Minv, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    computeForceAndHessian(q, qprev, qdot, F, H);
    qprev = q;
    q = q + params_.timeStep * qdot;
    qdot = qdot + params_.timeStep * Minv * F;
}

void implict_euler(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot, Eigen::SparseMatrix<double> &Minv, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
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
}

void implicit_midpoint(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot, Eigen::SparseMatrix<double> &Minv, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
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
}

void velocity_verlet(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot, Eigen::SparseMatrix<double> &Minv, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    qprev = q;
    q = q + params_.timeStep * qdot;
    computeForceAndHessian(q, qprev, qdot, F, H);
    qdot = qdot + params_.timeStep * Minv * F;
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
        explicit_euler(q, qprev, qdot, Minv, F, H);
        std::cout << "One step of Explicit Euler" << std::endl;
        break;
    }

    case SimParameters::TI_IMPLICIT_EULER:
    {
        implict_euler(q, qprev, qdot, Minv, F, H);
        std::cout << "One step of Implicit Euler" << std::endl;
        break;
    }

    case SimParameters::TI_IMPLICIT_MIDPOINT:
    {
        implicit_midpoint(q, qprev, qdot, Minv, F, H);
        std::cout << "One step of Implicit Midpoint" << std::endl;
        break;
    }

    case SimParameters::TI_VELOCITY_VERLET:
    {
        velocity_verlet(q, qprev, qdot, Minv, F, H);
        std::cout << "One step of Velocity Verlet" << std::endl;
        break;
    }

    default:
        std::cerr << "Invalid time integrator" << std::endl;
    }
}
