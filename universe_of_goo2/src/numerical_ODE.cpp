#include <iostream>
#include <functional>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

#include "forces_hessian.cpp"

void computeMassInverse(Eigen::SparseMatrix<double> &Minv)
{
    int ndofs = 2 * int(particles_.size());

    Minv.resize(ndofs, ndofs);
    Minv.setZero();

    std::vector<Eigen::Triplet<double>> Minvcoeffs;
    for (int i = 0; i < ndofs / 2; i++)
    {
        Minvcoeffs.push_back(Eigen::Triplet<double>(2 * i, 2 * i, 1.0 / getTotalParticleMass(i)));
        Minvcoeffs.push_back(Eigen::Triplet<double>(2 * i + 1, 2 * i + 1, 1.0 / getTotalParticleMass(i)));
    }

    Minv.setFromTriplets(Minvcoeffs.begin(), Minvcoeffs.end());
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

void penalty_method(const Eigen::VectorXd &q, Eigen::VectorXd &F)
{
    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
        {
            RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);
            Eigen::Vector2d pos1 = q.segment(2 * rod->p1, 2);
            Eigen::Vector2d pos2 = q.segment(2 * rod->p2, 2);

            Eigen::Vector2d dir = pos1 - pos2;
            double dist_squared = dir.squaredNorm();

            Eigen::Vector2d local_force = 4 * params_.penaltyStiffness * (dist_squared - pow(rod->length, 2)) * dir;

            F.segment(2 * rod->p1, 2) += -local_force;
            F.segment(2 * rod->p2, 2) += local_force;
        }
    }
}

void step_project_method()
{
}

void lagrange_multiplier_method()
{
}

void numericalIntegration(Eigen::VectorXd &q, Eigen::VectorXd &lambda, Eigen::VectorXd &qdot)
{
    Eigen::VectorXd F;
    Eigen::SparseMatrix<double> H;
    Eigen::SparseMatrix<double> Minv;

    computeMassInverse(Minv);

    Eigen::VectorXd oldq = q;

    q += params_.timeStep * qdot;
    computeForceAndHessian(q, oldq, F, H);

    // Modify the time integrator to handle constraints, based on the value of params_.constraintHandling
    switch (params_.constraintHandling)
    {
    case SimParameters::CH_PENALTY:
    {
        penalty_method(q, F);
        qdot += params_.timeStep * Minv * F;
        break;
    }

    case SimParameters::CH_STEPPROJECT:
    {
        qdot += params_.timeStep * Minv * F;
        step_project_method();
        break;
    }

    case SimParameters::CH_LAGRANGEMULT:
    {
        lagrange_multiplier_method();
        qdot += params_.timeStep * Minv * F;
        break;
    }

    default:
    {
        qdot += params_.timeStep * Minv * F;
        break;
    }
    }
}