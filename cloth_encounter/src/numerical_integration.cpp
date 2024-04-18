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
                         const Eigen::MatrixXi &F,
                         const std::vector<int> &pinnedVerts,
                         const int &clickedVertex,
                         const Eigen::Vector3d &mousePos)
{
    Eigen::MatrixXd Q_proj;
    for (int i = 0; i < params_.constraintIters; i++)
    {
        if (params_.pinEnabled)
        {
            compute_pin_constraint(Q, origQ, pinnedVerts);
            std::cout << "Project the pin constraints\n";
        }

        if (params_.stretchEnabled)
        {
            for (int j = 0; j < F.rows(); j++)
            {
                compute_stretch_constraint(Q, origQ, F.row(j));
            }
            std::cout << "Project the strech constraints\n";
        }

        if (params_.bendingEnabled)
        {
            compute_bending_constraint(Q);
            std::cout << "Project the bending constraints\n";
        }

        if (params_.pullingEnabled && clickedVertex != -1)
        {
            compute_pull_constraint(Q, clickedVertex, mousePos);
            std::cout << "Project the mouse drag constraints\n";
        }
    }
}

/*
    Position-based dynamics integration
*/
void numericalIntegration(Eigen::MatrixXd &Q,
                          Eigen::MatrixXd &Qdot,
                          const Eigen::MatrixXd &origQ,
                          const Eigen::MatrixXi &F,
                          const std::vector<int> &pinnedVerts,
                          const int &clickedVertex,
                          const Eigen::Vector3d &mousePos)
{
    Eigen::MatrixXd Q_old = Q;
    Q = Q + params_.timeStep * Qdot;

    project_constraints(Q, origQ, F, pinnedVerts, clickedVertex, mousePos);

    Eigen::MatrixXd force;
    computeForce(Q, force);

    Qdot = (Q - Q_old) / params_.timeStep;
    Qdot = Qdot + params_.timeStep * force;

    std::cout << "Numerically integrate to update the position and velocity\n";
}