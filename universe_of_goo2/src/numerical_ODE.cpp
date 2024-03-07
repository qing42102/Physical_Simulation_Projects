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

Eigen::SparseMatrix<double> compute_constraint_deriv(const Eigen::VectorXd &q, const int &num_rigid_rods)
{
    Eigen::SparseMatrix<double> constrain_deriv(num_rigid_rods, q.size());
    std::vector<Eigen::Triplet<double>> triplet_list;
    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
        {
            RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);
            Eigen::Vector2d pos1 = q.segment(2 * rod->p1, 2);
            Eigen::Vector2d pos2 = q.segment(2 * rod->p2, 2);

            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p1, 2 * (pos1 - pos2)[0]));
            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p1 + 1, 2 * (pos1 - pos2)[1]));

            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p2, -2 * (pos1 - pos2)[0]));
            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p2 + 1, -2 * (pos1 - pos2)[1]));
        }
    }
    constrain_deriv.setFromTriplets(triplet_list.begin(), triplet_list.end());

    return constrain_deriv;
}

Eigen::VectorXd compute_constraint(const Eigen::VectorXd &q, const int &num_rigid_rods)
{
    Eigen::VectorXd constraint(num_rigid_rods);
    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
        {
            RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);
            Eigen::Vector2d pos1 = q.segment(2 * rod->p1, 2);
            Eigen::Vector2d pos2 = q.segment(2 * rod->p2, 2);

            constraint(i) = (pos1 - pos2).squaredNorm() - pow(rod->length, 2);
        }
    }
    return constraint;
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

void step_project_method(const Eigen::VectorXd &unconstrain_q,
                         const Eigen::VectorXd &unconstrain_qdot,
                         const Eigen::SparseMatrix<double> &Minv,
                         Eigen::VectorXd &constrain_q,
                         Eigen::VectorXd &constrain_qdot)
{
    int num_rigid_rods = 0;
    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
        {
            num_rigid_rods++;
        }
    }

    Eigen::SparseMatrix<double> Minv_modified = Minv;
    for (uint i = 0; i < particles_.size(); i++)
    {
        if (particles_[i].fixed)
        {
            Minv_modified.coeffRef(2 * i, 2 * i) = 0;
        }
    }

    // Lambda function for the step and project method and its derivative
    auto func = [&](Eigen::VectorXd state) -> Eigen::VectorXd
    {
        Eigen::VectorXd q = state.segment(0, 2 * particles_.size());
        Eigen::VectorXd lambda = state.segment(2 * particles_.size(), num_rigid_rods);

        Eigen::VectorXd constraint = compute_constraint(q, num_rigid_rods);
        Eigen::SparseMatrix<double> constraint_deriv = compute_constraint_deriv(q, num_rigid_rods);

        Eigen::VectorXd constraint_multiplier;
        constraint_multiplier.resize(q.size());
        constraint_multiplier.setZero();
        for (uint i = 0; i < connectors_.size(); i++)
        {
            if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
            {
                RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);

                constraint_multiplier.segment(2 * rod->p1, 2) += lambda(i) * constraint_deriv.block(i, 2 * rod->p1, 1, 2).transpose();
                constraint_multiplier.segment(2 * rod->p2, 2) += lambda(i) * constraint_deriv.block(i, 2 * rod->p2, 1, 2).transpose();
            }
        }
        constraint_multiplier = Minv_modified * constraint_multiplier;

        Eigen::VectorXd f(q.size() + num_rigid_rods);
        f << (q - unconstrain_q) + constraint_multiplier, constraint;
        return f;
    };

    auto deriv = [&](Eigen::VectorXd state) -> Eigen::SparseMatrix<double>
    {
        Eigen::VectorXd q = state.segment(0, 2 * particles_.size());
        Eigen::VectorXd lambda = state.segment(2 * particles_.size(), num_rigid_rods);

        Eigen::SparseMatrix<double> constraint_deriv = compute_constraint_deriv(q, num_rigid_rods);

        Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(q.size(), q.size()).sparseView();

        std::vector<Eigen::Triplet<double>> triplet_list;
        for (uint i = 0; i < connectors_.size(); i++)
        {
            if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
            {
                RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);

                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p1, 2 * rod->p1, 1 + 2 * lambda(i)));
                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p1 + 1, 2 * rod->p1 + 1, 2 * lambda(i)));
                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p2, 2 * rod->p2, 1 - 2 * lambda(i)));
                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p2 + 1, 2 * rod->p2 + 1, -2 * lambda(i)));

                triplet_list.push_back(Eigen::Triplet<double>(q.size() + i, 2 * rod->p1, Minv_modified.coeff(2 * rod->p1, 2 * rod->p1) * constraint_deriv.coeff(i, 2 * rod->p1)));
                triplet_list.push_back(Eigen::Triplet<double>(q.size() + i, 2 * rod->p1 + 1, Minv_modified.coeff(2 * rod->p1 + 1, 2 * rod->p1 + 1) * constraint_deriv.coeff(i, 2 * rod->p1 + 1)));
                triplet_list.push_back(Eigen::Triplet<double>(q.size() + i, 2 * rod->p2, Minv_modified.coeff(2 * rod->p2, 2 * rod->p2) * constraint_deriv.coeff(i, 2 * rod->p2)));
                triplet_list.push_back(Eigen::Triplet<double>(q.size() + i, 2 * rod->p2 + 1, Minv_modified.coeff(2 * rod->p2 + 1, 2 * rod->p2 + 1) * constraint_deriv.coeff(i, 2 * rod->p2 + 1)));

                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p1, q.size() + i, constraint_deriv.coeff(i, 2 * rod->p1)));
                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p1 + 1, q.size() + i, constraint_deriv.coeff(i, 2 * rod->p1 + 1)));
                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p2, q.size() + i, constraint_deriv.coeff(i, 2 * rod->p2)));
                triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p2 + 1, q.size() + i, constraint_deriv.coeff(i, 2 * rod->p2 + 1)));
            }
        }
        Eigen::SparseMatrix<double> df(q.size() + num_rigid_rods, q.size() + num_rigid_rods);
        df.setFromTriplets(triplet_list.begin(), triplet_list.end());

        return df;
    };

    Eigen::VectorXd initial_state(2 * particles_.size() + num_rigid_rods);
    initial_state.segment(0, 2 * particles_.size()) = unconstrain_q;
    initial_state.segment(2 * particles_.size(), num_rigid_rods) = Eigen::VectorXd::Zero(num_rigid_rods);

    Eigen::VectorXd state = newton_method(func, deriv, initial_state);
    constrain_q = state.segment(0, 2 * particles_.size());
    constrain_qdot = unconstrain_qdot + (constrain_q - unconstrain_q) / params_.timeStep;
}

void lagrange_multiplier_method(const Eigen::VectorXd &q,
                                Eigen::VectorXd &qdot,
                                Eigen::VectorXd &lambda,
                                const Eigen::SparseMatrix<double> &Minv,
                                const Eigen::VectorXd &F)
{
    auto func = [&](Eigen::VectorXd lambda) -> Eigen::VectorXd
    {
        Eigen::VectorXd input = q + params_.timeStep * qdot + (pow(params_.timeStep, 2) * Minv) * (F + compute_constraint_deriv(q, lambda.size()).transpose() * lambda);
        return compute_constraint(input, lambda.size());
    };

    auto deriv = [&](Eigen::VectorXd lambda) -> Eigen::SparseMatrix<double>
    {
        Eigen::VectorXd input = q + params_.timeStep * qdot + (pow(params_.timeStep, 2) * Minv) * (F + compute_constraint_deriv(q, lambda.size()).transpose() * lambda);

        // Chain rule for the derivative of the constraint
        Eigen::SparseMatrix<double> dx = pow(params_.timeStep, 2) * Minv * compute_constraint_deriv(q, lambda.size()).transpose();
        Eigen::SparseMatrix<double> dgdx = 2 * compute_constraint_deriv(input, lambda.size());
        Eigen::SparseMatrix<double> df = dgdx * dx;

        return df;
    };

    lambda = newton_method(func, deriv, lambda);
    qdot = qdot + params_.timeStep * Minv * (F + compute_constraint_deriv(q, lambda.size()).transpose() * lambda);
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

        Eigen::VectorXd constrain_q;
        Eigen::VectorXd constrain_qdot;
        step_project_method(q, qdot, Minv, constrain_q, constrain_qdot);
        break;
    }

    case SimParameters::CH_LAGRANGEMULT:
    {
        lagrange_multiplier_method(q, qdot, lambda, Minv, F);
        break;
    }

    default:
    {
        qdot += params_.timeStep * Minv * F;
        break;
    }
    }
}