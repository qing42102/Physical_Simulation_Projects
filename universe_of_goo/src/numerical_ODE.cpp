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
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(2 * particles_.size());
    for (uint i = 0; i < particles_.size(); i++)
    {
        triplet_list.push_back(Eigen::Triplet<double>(2 * i, 2 * i, 1.0 / particles_[i].mass));
        triplet_list.push_back(Eigen::Triplet<double>(2 * i + 1, 2 * i + 1, 1.0 / particles_[i].mass));
    }

    Minv.setFromTriplets(triplet_list.begin(), triplet_list.end());
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
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
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

void explicit_euler(Eigen::VectorXd &q,
                    Eigen::VectorXd &qprev,
                    Eigen::VectorXd &qdot,
                    const Eigen::SparseMatrix<double> &Minv)
{
    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    computeForceAndHessian(q, qprev, qdot, F, H);
    qprev = q;
    q = q + params_.timeStep * qdot;
    qdot = qdot + params_.timeStep * Minv * F;
}

void implicit_euler(Eigen::VectorXd &q,
                    Eigen::VectorXd &qprev,
                    Eigen::VectorXd &qdot,
                    const Eigen::SparseMatrix<double> &Minv)
{
    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    qprev = q;
    // Lambda function for the implicit Euler method and its derivative
    auto func = [&](Eigen::VectorXd q) -> Eigen::VectorXd
    {
        computeForceAndHessian(q, qprev, qdot, F, H);
        return q - qprev - params_.timeStep * (qdot + params_.timeStep * Minv * F);
    };

    // F and H is shared between the function and its derivative
    auto deriv = [&](Eigen::VectorXd q) -> Eigen::SparseMatrix<double>
    {
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
                       const Eigen::SparseMatrix<double> &Minv)
{
    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    Eigen::VectorXd avg_q;

    qprev = q;
    // Lambda function for the implicit midpoint method and its derivative
    auto func = [&](Eigen::VectorXd q) -> Eigen::VectorXd
    {
        avg_q = (q + qprev) / 2;
        computeForceAndHessian(avg_q, qprev, qdot, F, H);
        return q - qprev - params_.timeStep * (2 * qdot + params_.timeStep * Minv * F) / 2;
    };

    // F and H is shared between the function and its derivative
    auto deriv = [&](Eigen::VectorXd q) -> Eigen::SparseMatrix<double>
    {
        Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(2 * particles_.size(), 2 * particles_.size()).sparseView();
        return sparse_I - 0.25 * pow(params_.timeStep, 2) * Minv * H;
    };

    q = newton_method(func, deriv, q);
    avg_q = (q + qprev) / 2;
    computeForceAndHessian(avg_q, qprev, qdot, F, H);
    qdot = qdot + params_.timeStep * Minv * F;
}

void velocity_verlet(Eigen::VectorXd &q,
                     Eigen::VectorXd &qprev,
                     Eigen::VectorXd &qdot,
                     const Eigen::SparseMatrix<double> &Minv)
{
    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    qprev = q;
    q = q + params_.timeStep * qdot;
    computeForceAndHessian(q, qprev, qdot, F, H);
    qdot = qdot + params_.timeStep * Minv * F;
}

/*
    Runge Kutta Felhberg method for solving a system of ODEs
*/
void Runge_Kutta_45(Eigen::VectorXd &q,
                    Eigen::VectorXd &qprev,
                    Eigen::VectorXd &qdot,
                    const Eigen::SparseMatrix<double> &Minv,
                    const double tolerance)
{
    Eigen::VectorXd F(2 * particles_.size());
    Eigen::SparseMatrix<double> H(2 * particles_.size(), 2 * particles_.size());

    // Butcher table for the Runge Kutta Felhberg method
    double B_2_1 = 0.25;
    double B_3_1 = 3.0 / 32.0, B_3_2 = 9.0 / 32.0;
    double B_4_1 = 1932.0 / 2197.0, B_4_2 = -7200.0 / 2197.0, B_4_3 = 7296.0 / 2197.0;
    double B_5_1 = 439.0 / 216.0, B_5_2 = -8.0, B_5_3 = 3680.0 / 513.0, B_5_4 = -845.0 / 4104.0;
    double B_6_1 = -8.0 / 27.0, B_6_2 = 2.0, B_6_3 = -3544.0 / 2565.0, B_6_4 = 1859.0 / 4104.0, B_6_5 = -11.0 / 40.0;

    double C_1 = 16.0 / 135.0, C_3 = 6656.0 / 12825.0, C_4 = 28561.0 / 56430.0, C_5 = -9.0 / 50.0, C_6 = 2.0 / 55.0;

    double CT_1 = -1.0 / 360.0, CT_3 = 128.0 / 4275.0, CT_4 = 2197.0 / 75240.0, CT_5 = -1.0 / 50.0, CT_6 = -2.0 / 55.0;

    // Compute the k1, k2, k3, k4, k5, k6 vectors
    Eigen::VectorXd k1_q = params_.timeStep * qdot;
    computeForceAndHessian(q, qprev, qdot, F, H);
    Eigen::VectorXd k1_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k2_q = params_.timeStep * (qdot + B_2_1 * k1_qdot);
    Eigen::VectorXd q_new = q + B_2_1 * k1_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k2_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k3_q = params_.timeStep * (qdot + B_3_1 * k1_qdot + B_3_2 * k2_qdot);
    q_new = q + B_3_1 * k1_q + B_3_2 * k2_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k3_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k4_q = params_.timeStep * (qdot + B_4_1 * k1_qdot + B_4_2 * k2_qdot + B_4_3 * k3_qdot);
    q_new = q + B_4_1 * k1_q + B_4_2 * k2_q + B_4_3 * k3_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k4_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k5_q = params_.timeStep * (qdot + B_5_1 * k1_qdot + B_5_2 * k2_qdot + B_5_3 * k3_qdot + B_5_4 * k4_qdot);
    q_new = q + B_5_1 * k1_q + B_5_2 * k2_q + B_5_3 * k3_q + B_5_4 * k4_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k5_qdot = params_.timeStep * Minv * F;

    Eigen::VectorXd k6_q = params_.timeStep * (qdot + B_6_1 * k1_qdot + B_6_2 * k2_qdot + B_6_3 * k3_qdot + B_6_4 * k4_qdot + B_6_5 * k5_qdot);
    q_new = q + B_6_1 * k1_q + B_6_2 * k2_q + B_6_3 * k3_q + B_6_4 * k4_q + B_6_5 * k5_q;
    computeForceAndHessian(q_new, qprev, qdot, F, H);
    Eigen::VectorXd k6_qdot = params_.timeStep * Minv * F;

    qprev = q;

    // Compute the new q and qdot vectors
    q = q + C_1 * k1_q + C_3 * k3_q + C_4 * k4_q + C_5 * k5_q + C_6 * k6_q;
    qdot = qdot + C_1 * k1_qdot + C_3 * k3_qdot + C_4 * k4_qdot + C_5 * k5_qdot + C_6 * k6_qdot;

    // Compute the error
    Eigen::VectorXd error_q = CT_1 * k1_q + CT_3 * k3_q + CT_4 * k4_q + CT_5 * k5_q + CT_6 * k6_q;
    Eigen::VectorXd error_qdot = CT_1 * k1_qdot + CT_3 * k3_qdot + CT_4 * k4_qdot + CT_5 * k5_qdot + CT_6 * k6_qdot;
    double error = sqrt(error_q.squaredNorm() + error_qdot.squaredNorm());

    // Adaptive time step
    double new_time_step = params_.timeStep * 0.9 * pow(tolerance / error, 0.2);
    if (new_time_step < 0.01)
    {
        params_.timeStep = new_time_step;
    }

    std::cout << "Adaptive time step: " << params_.timeStep << "\n";
}

void numericalIntegration(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot)
{
    if (q.size() == 0)
    {
        return;
    }

    Eigen::SparseMatrix<double> Minv(2 * particles_.size(), 2 * particles_.size());
    computeMassInverse(Minv);

    // Perform one step of time integration, using the method in params_.integrator
    switch (params_.integrator)
    {
    case SimParameters::TI_EXPLICIT_EULER:
    {
        explicit_euler(q, qprev, qdot, Minv);
        std::cout << "One step of Explicit Euler\n";
        break;
    }

    case SimParameters::TI_IMPLICIT_EULER:
    {
        implicit_euler(q, qprev, qdot, Minv);
        std::cout << "One step of Implicit Euler\n";
        break;
    }

    case SimParameters::TI_IMPLICIT_MIDPOINT:
    {
        implicit_midpoint(q, qprev, qdot, Minv);
        std::cout << "One step of Implicit Midpoint\n";
        break;
    }

    case SimParameters::TI_VELOCITY_VERLET:
    {
        velocity_verlet(q, qprev, qdot, Minv);
        std::cout << "One step of Velocity Verlet\n";
        break;
    }

    case SimParameters::TI_RK_45:
    {
        Runge_Kutta_45(q, qprev, qdot, Minv, 1e-6);
        std::cout << "One step of Runge Kutta 45\n";
        break;
    }

    default:
        std::cerr << "Invalid time integrator\n";
    }
}
