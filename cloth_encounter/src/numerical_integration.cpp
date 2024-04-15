#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "constraint.cpp"
#include "forces.cpp"

/*
    Projects the coordinates of the vertices to satisfy the constraints
*/
void project_constraints(Eigen::MatrixXd &Q,
                         const Eigen::MatrixXd &origQ,
                         const std::vector<int> &pinnedVerts)
{
    Eigen::MatrixXd Q_proj;
    for (int i = 0; i < params_.constraintIters; i++)
    {
        Q_proj = compute_pin_constraint(Q, origQ, pinnedVerts);
        Q = params_.pinWeight * Q_proj + (1 - params_.pinWeight) * Q;

        Q_proj = compute_stretch_constraint(Q);
        Q = params_.stretchWeight * Q_proj + (1 - params_.stretchWeight) * Q;

        Q_proj = compute_bending_constraint(Q);
        Q = params_.bendingWeight * Q_proj + (1 - params_.bendingWeight) * Q;

        Q_proj = compute_pull_constraint(Q);
        Q = params_.pullingWeight * Q_proj + (1 - params_.pullingWeight) * Q;
    }
}

/*
    Position-based dynamics integration
*/
void numericalIntegration(Eigen::MatrixXd &Q,
                          Eigen::MatrixXd &Qdot,
                          const Eigen::MatrixXd &origQ,
                          const std::vector<int> &pinnedVerts)
{
    Eigen::MatrixXd Q_old = Q;
    Q = Q + params_.timeStep * Qdot;

    project_constraints(Q, origQ, pinnedVerts);

    Eigen::MatrixXd force;
    computeForce(Q, force);

    Qdot = (Q - Q_old) / params_.timeStep;
    Qdot = Qdot + params_.timeStep * force;

    std::cout << "Numerically integrate to update the position and velocity\n";
}