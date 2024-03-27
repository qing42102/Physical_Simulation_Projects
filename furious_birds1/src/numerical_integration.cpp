#include <iostream>
#include <functional>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "forces.cpp"
#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"

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

void numericalIntegration()
{
}