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

void explicit_euler(Eigen::VectorXd &q,
                    Eigen::VectorXd &qprev,
                    Eigen::VectorXd &qdot,
                    Eigen::SparseMatrix<double> &Minv,
                    Eigen::VectorXd &F,
                    Eigen::SparseMatrix<double> &H)
{
    computeForceAndHessian(q, qprev, qdot, F, H);
    qprev = q;
    q = q + params_.timeStep * qdot;
    qdot = qdot + params_.timeStep * Minv * F;
}

void implicit_euler(Eigen::VectorXd &q,
                    Eigen::VectorXd &qprev,
                    Eigen::VectorXd &qdot,
                    Eigen::SparseMatrix<double> &Minv,
                    Eigen::VectorXd &F,
                    Eigen::SparseMatrix<double> &H)
{
    qprev = q;
    // Lambda function for the implicit Euler method and its derivative
    auto func = [&](Eigen::VectorXd q) -> Eigen::VectorXd
    {
        computeForceAndHessian(q, qprev, qdot, F, H);
        return q - qprev - params_.timeStep * (qdot + params_.timeStep * Minv * F);
    };

    auto deriv = [&](Eigen::VectorXd q) -> Eigen::SparseMatrix<double>
    {
        computeForceAndHessian(q, qprev, qdot, F, H);
        Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(2 * particles_.size(), 2 * particles_.size()).sparseView();
        return sparse_I - pow(params_.timeStep, 2) * Minv * H;
    };

    q = newton_method(func, deriv, q);
    computeForceAndHessian(q, qprev, qdot, F, H);
    qdot = qdot + params_.timeStep * Minv * F;
}

void implicit_midpoint(Eigen::VectorXd &q,
                       Eigen::VectorXd &qprev,
                       Eigen::VectorXd &qdot,
                       Eigen::SparseMatrix<double> &Minv,
                       Eigen::VectorXd &F,
                       Eigen::SparseMatrix<double> &H)
{
    qprev = q;
    // Lambda function for the implicit midpoint method and its derivative
    auto func = [&](Eigen::VectorXd q) -> Eigen::VectorXd
    {
        Eigen::VectorXd avg_q = (q + qprev) / 2;
        computeForceAndHessian(avg_q, qprev, qdot, F, H);
        return q - qprev - params_.timeStep * (2 * qdot + params_.timeStep * Minv * F) / 2;
    };

    auto deriv = [&](Eigen::VectorXd q) -> Eigen::SparseMatrix<double>
    {
        Eigen::VectorXd avg_q = (q + qprev) / 2;
        computeForceAndHessian(avg_q, qprev, qdot, F, H);
        Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(2 * particles_.size(), 2 * particles_.size()).sparseView();
        return sparse_I - 0.25 * pow(params_.timeStep, 2) * Minv * H;
    };

    q = newton_method(func, deriv, q);
    Eigen::VectorXd avg_q = (q + qprev) / 2;
    computeForceAndHessian(avg_q, qprev, qdot, F, H);
    qdot = qdot + params_.timeStep * Minv * F;
}

void velocity_verlet(Eigen::VectorXd &q,
                     Eigen::VectorXd &qprev,
                     Eigen::VectorXd &qdot,
                     Eigen::SparseMatrix<double> &Minv,
                     Eigen::VectorXd &F,
                     Eigen::SparseMatrix<double> &H)
{
    qprev = q;
    q = q + params_.timeStep * qdot;
    computeForceAndHessian(q, qprev, qdot, F, H);
    qdot = qdot + params_.timeStep * Minv * F;
}

void Runge_Kutta_45(Eigen::VectorXd &q,
                    Eigen::VectorXd &qprev,
                    Eigen::VectorXd &qdot,
                    Eigen::SparseMatrix<double> &Minv,
                    Eigen::VectorXd &F,
                    Eigen::SparseMatrix<double> &H,
                    double tolerance)
{
    // Runge Kutta Felhberg method for solving a system of ODEs

    // Compute the k1, k2, k3, k4, k5, k6 vectors
    Eigen::VectorXd k1_q = params_.timeStep * qdot;
    computeForceAndHessian(q, qprev, qdot, F, H);
    Eigen::VectorXd k1_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k2_q = params_.timeStep * (qdot + 0.25 * k1_qdot);
    Eigen::VectorXd q_new = q + 0.25 * k1_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k2_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k3_q = params_.timeStep * (qdot + 3.0 / 32.0 * k1_qdot + 9.0 / 32.0 * k2_qdot);
    q_new = q + 3.0 / 32.0 * k1_q + 9.0 / 32.0 * k2_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k3_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k4_q = params_.timeStep * (qdot + 1932.0 / 2197.0 * k1_qdot - 7200.0 / 2197.0 * k2_qdot + 7296.0 / 2197.0 * k3_qdot);
    q_new = q + 1932.0 / 2197.0 * k1_q - 7200.0 / 2197.0 * k2_q + 7296.0 / 2197.0 * k3_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k4_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k5_q = params_.timeStep * (qdot + 439.0 / 216.0 * k1_qdot - 8.0 * k2_qdot + 3680.0 / 513.0 * k3_qdot - 845.0 / 4104.0 * k4_qdot);
    q_new = q + 439.0 / 216.0 * k1_q - 8.0 * k2_q + 3680.0 / 513.0 * k3_q - 845.0 / 4104.0 * k4_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k5_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k6_q = params_.timeStep * (qdot - 8.0 / 27.0 * k1_qdot + 2.0 * k2_qdot - 3544.0 / 2565.0 * k3_qdot + 1859.0 / 4104.0 * k4_qdot - 11.0 / 40.0 * k5_qdot);
    q_new = q - 8.0 / 27.0 * k1_q + 2.0 * k2_q - 3544.0 / 2565.0 * k3_q + 1859.0 / 4104.0 * k4_q - 11.0 / 40.0 * k5_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k6_qdot = params_.timeStep * Minv * F;

    qprev = q;

    // Compute the new q and qdot vectors
    q = q + 16.0 / 135.0 * k1_q + 6656.0 / 12825.0 * k3_q + 28561.0 / 56430.0 * k4_q - 9.0 / 50.0 * k5_q + 2.0 / 55.0 * k6_q;
    qdot = qdot + 16.0 / 135.0 * k1_qdot + 6656.0 / 12825.0 * k3_qdot + 28561.0 / 56430.0 * k4_qdot - 9.0 / 50.0 * k5_qdot + 2.0 / 55.0 * k6_qdot;

    // Compute the error
    Eigen::VectorXd error_q = -1.0 / 360.0 * k1_q + 128.0 / 4275.0 * k3_q + 2197.0 / 75240.0 * k4_q - 1.0 / 50.0 * k5_q - 2.0 / 55.0 * k6_q;
    Eigen::VectorXd error_qdot = -1.0 / 360.0 * k1_qdot + 128.0 / 4275.0 * k3_qdot + 2197.0 / 75240.0 * k4_qdot - 1.0 / 50.0 * k5_qdot - 2.0 / 55.0 * k6_qdot;
    double error = sqrt(error_q.squaredNorm() + error_qdot.squaredNorm());

    // Adaptive time step
    double new_time_step = params_.timeStep * 0.9 * pow(tolerance / error, 0.2);
    if (new_time_step < 0.01)
    {
        params_.timeStep = new_time_step;
    }

    std::cout << "Adaptive time step: " << params_.timeStep << std::endl;
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
        implicit_euler(q, qprev, qdot, Minv, F, H);
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

    case SimParameters::TI_RK_45:
    {
        Runge_Kutta_45(q, qprev, qdot, Minv, F, H, 1e-6);
        std::cout << "One step of Runge Kutta 45" << std::endl;
        break;
    }

    default:
        std::cerr << "Invalid time integrator" << std::endl;
    }
}
