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
    Compute the constraint derivatives, 2 * (pos1 - pos2)
    The dimension of output is the number of constraints x the number of degrees of freedom
*/
Eigen::SparseMatrix<double> compute_constraint_deriv(const Eigen::VectorXd &q)
{
    Eigen::SparseMatrix<double> constrain_deriv(connectors_.size(), q.size());
    std::vector<Eigen::Triplet<double>> triplet_list;
    for (uint i = 0; i < connectors_.size(); i++)
    {
        // Connectors that are not rigid rods have no constraint
        if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
        {
            RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);
            Eigen::Vector2d pos1 = q.segment(2 * rod->p1, 2);
            Eigen::Vector2d pos2 = q.segment(2 * rod->p2, 2);

            Eigen::Vector2d dir = pos1 - pos2;

            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p1, 2 * dir[0]));
            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p1 + 1, 2 * dir[1]));

            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p2, -2 * dir[0]));
            triplet_list.push_back(Eigen::Triplet<double>(i, 2 * rod->p2 + 1, -2 * dir[1]));
        }
    }
    constrain_deriv.setFromTriplets(triplet_list.begin(), triplet_list.end());

    return constrain_deriv;
}

/*
    Compute the constraint, ||pos1 - pos2||^2 - length^2
    The dimension of output is the number of constraints
 */
Eigen::VectorXd compute_constraint(const Eigen::VectorXd &q)
{
    Eigen::VectorXd constraint(connectors_.size());
    constraint.setZero();
    for (uint i = 0; i < connectors_.size(); i++)
    {
        // Connectors that are not rigid rods have no constraint
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

/*
    Compute the penalty method for the constraints
    The penalty force is F = -4 * k * (||pos1 - pos2||^2 - length^2) * (pos1 - pos2)
*/
void penalty_method(const Eigen::VectorXd &q, Eigen::VectorXd &F)
{
    for (uint i = 0; i < connectors_.size(); i++)
    {
        // Connectors that are not rigid rods have no constraint
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

/*
    Compute the step and project method for the constraints
    Take an unconstrained step using the time integrator
    Then project the new position to satisfy the constraints

    f(\mathbf{q}) =
    \begin{bmatrix}
        (\mathbf{q} - \tilde{\mathbf{q}}) + \sum_{i=1}^m \lambda_i \mathbf{M}^{-1} [d g_i(\mathbf{q})]^T \\
        \mathbf{g}(\mathbf{q})
    \end{bmatrix} = 0

    df(\mathbf{q}) =
    \begin{bmatrix}
        \mathbf{I} + \sum_i \lambda_i \mathbf{M}^{-1} [H g_i(\mathbf{q})] & \mathbf{M}^{-1} [d \mathbf{g}(\mathbf{q})]^T \\
        [d \mathbf{g}(\mathbf{q})] & 0
    \end{bmatrix}

    where $[H g_i(\mathbf{q})] = 2 (\mathbf{S}_a - \mathbf{S}_b)$.
*/
void step_project_method(const Eigen::VectorXd &unconstrain_q,
                         const Eigen::VectorXd &unconstrain_qdot,
                         const Eigen::SparseMatrix<double> &Minv,
                         Eigen::VectorXd &constrain_q,
                         Eigen::VectorXd &constrain_qdot)
{
    // Modify the mass matrix to set the mass to 0 for fixed particles
    Eigen::SparseMatrix<double> Minv_modified = Minv;
    for (uint i = 0; i < particles_.size(); i++)
    {
        if (particles_[i].fixed)
        {
            Minv_modified.coeffRef(2 * i, 2 * i) = 0;
            Minv_modified.coeffRef(2 * i + 1, 2 * i + 1) = 0;
        }
    }

    Eigen::SparseMatrix<double> sparse_I = Eigen::MatrixXd::Identity(unconstrain_q.size(), unconstrain_q.size()).sparseView();

    // Lambda function for the step and project method and its derivative
    auto func = [&](Eigen::VectorXd state) -> Eigen::VectorXd
    {
        Eigen::VectorXd q = state.segment(0, 2 * particles_.size());
        Eigen::VectorXd lambda = state.segment(2 * particles_.size(), connectors_.size());

        Eigen::VectorXd constraint = compute_constraint(q);
        Eigen::SparseMatrix<double> constraint_deriv = compute_constraint_deriv(q);

        Eigen::VectorXd constraint_multiplier = Minv_modified * constraint_deriv.transpose() * lambda;

        Eigen::VectorXd f(q.size() + connectors_.size());
        f << (q - unconstrain_q) + constraint_multiplier, constraint;
        return f;
    };

    auto deriv = [&](Eigen::VectorXd state) -> Eigen::SparseMatrix<double>
    {
        Eigen::VectorXd q = state.segment(0, 2 * particles_.size());
        Eigen::VectorXd lambda = state.segment(2 * particles_.size(), connectors_.size());

        Eigen::VectorXd constraint = compute_constraint(q);
        Eigen::SparseMatrix<double> constraint_deriv = compute_constraint_deriv(q);

        // Upper left block of the Jacobian
        Eigen::SparseMatrix<double> upper_left(q.size(), q.size());
        std::vector<Eigen::Triplet<double>> triplet_list;
        for (uint i = 0; i < connectors_.size(); i++)
        {
            // Connectors that are not rigid rods have no constraint
            if (connectors_[i]->getType() == SimParameters::CT_RIGIDROD)
            {
                RigidRod *rod = dynamic_cast<RigidRod *>(connectors_[i]);

                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        // Derivative of the constraint with respect to q
                        // Since it's a diagonal matrix, the values only exists when j = k
                        if (j == k)
                        {
                            triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p1 + j, 2 * rod->p1 + k, 2 * lambda(i)));
                            triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p2 + j, 2 * rod->p2 + k, 2 * lambda(i)));
                            triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p1 + j, 2 * rod->p2 + k, -2 * lambda(i)));
                            triplet_list.push_back(Eigen::Triplet<double>(2 * rod->p2 + j, 2 * rod->p1 + k, -2 * lambda(i)));
                        }
                    }
                }
            }
        }
        upper_left.setFromTriplets(triplet_list.begin(), triplet_list.end());
        upper_left = Minv_modified * upper_left + sparse_I;

        // Lower left block of the Jacobian
        Eigen::SparseMatrix<double> lower_left = constraint_deriv;

        // Upper right block of the Jacobian
        Eigen::SparseMatrix<double> upper_right = Minv_modified * constraint_deriv.transpose();

        Eigen::SparseMatrix<double> df(q.size() + connectors_.size(), q.size() + connectors_.size());
        triplet_list.clear();
        triplet_list.reserve(upper_left.nonZeros() + upper_right.nonZeros() + lower_left.nonZeros());
        // Collect triplets from upper left matrix
        for (int i = 0; i < upper_left.outerSize(); i++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(upper_left, i); it; ++it)
            {
                triplet_list.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
            }
        }

        // Collect triplets from upper right matrix
        for (int i = 0; i < upper_right.outerSize(); i++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(upper_right, i); it; ++it)
            {
                triplet_list.push_back(Eigen::Triplet<double>(it.row(), it.col() + upper_left.cols(), it.value()));
            }
        }

        // Collect triplets from lower left matrix
        for (int i = 0; i < lower_left.outerSize(); i++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(lower_left, i); it; ++it)
            {
                triplet_list.push_back(Eigen::Triplet<double>(it.row() + upper_left.rows(), it.col(), it.value()));
            }
        }
        df.setFromTriplets(triplet_list.begin(), triplet_list.end());

        return df;
    };

    // State vector is concatenation of q and lambda
    // Initial guess for lambda is 0 and for q is the unconstrained q
    Eigen::VectorXd initial_state(2 * particles_.size() + connectors_.size());
    initial_state.segment(0, 2 * particles_.size()) = unconstrain_q;
    initial_state.segment(2 * particles_.size(), connectors_.size()) = Eigen::VectorXd::Zero(connectors_.size());

    Eigen::VectorXd state = newton_method(func, deriv, initial_state);
    constrain_q = state.segment(0, 2 * particles_.size());
    constrain_qdot = unconstrain_qdot + (constrain_q - unconstrain_q) / params_.timeStep;
}

/*
    Compute the Lagrange multiplier method for the constraints
    Construct the Lagrangian and solve for the equations of motion using Newton's method

    f(\lambda^{i+1}) = \mathbf{g}\left(\mathbf{q}^{i+1} + h M^{-1}(\mathbf{p}^i)^T + h^2 M^{-1} F \mathbf{q}^{i+1}) + h^2 M^{-1} [d \mathbf{g}(\mathbf{q}^{i+1})]^T \lambda^{i+1} \right) = 0
*/
void lagrange_multiplier_method(const Eigen::VectorXd &q,
                                Eigen::VectorXd &qdot,
                                Eigen::VectorXd &lambda,
                                const Eigen::SparseMatrix<double> &Minv,
                                const Eigen::VectorXd &F)
{
    // Precompute some terms for the lambda functions
    Eigen::SparseMatrix<double> constraint_deriv = compute_constraint_deriv(q);
    Eigen::VectorXd Min_F = Minv * F;
    Eigen::VectorXd term1 = q + params_.timeStep * qdot + pow(params_.timeStep, 2) * Min_F;
    Eigen::SparseMatrix<double> term2 = pow(params_.timeStep, 2) * Minv * constraint_deriv.transpose();

    auto func = [&](Eigen::VectorXd lambda) -> Eigen::VectorXd
    {
        Eigen::VectorXd input = term1 + term2 * lambda;
        return compute_constraint(input);
    };

    auto deriv = [&](Eigen::VectorXd lambda) -> Eigen::SparseMatrix<double>
    {
        // Chain rule for the derivative of the constraint
        Eigen::VectorXd input = term1 + term2 * lambda;
        Eigen::SparseMatrix<double> dgdx = compute_constraint_deriv(input);
        Eigen::SparseMatrix<double> df = dgdx * term2;

        return df;
    };

    lambda = newton_method(func, deriv, lambda);
    qdot = qdot + params_.timeStep * Min_F + term2 / params_.timeStep * lambda;
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

        std::cout << "Penalty method\n";
        break;
    }

    case SimParameters::CH_STEPPROJECT:
    {
        qdot += params_.timeStep * Minv * F;

        Eigen::VectorXd constrain_q;
        Eigen::VectorXd constrain_qdot;
        step_project_method(q, qdot, Minv, constrain_q, constrain_qdot);
        q = constrain_q;
        qdot = constrain_qdot;

        std::cout << "Step and project method\n";
        break;
    }

    case SimParameters::CH_LAGRANGEMULT:
    {
        lagrange_multiplier_method(q, qdot, lambda, Minv, F);

        std::cout << "Lagrange multiplier method\n";
        break;
    }

    default:
    {
        qdot += params_.timeStep * Minv * F;
        break;
    }
    }
}