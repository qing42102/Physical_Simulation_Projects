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
                         const Eigen::Vector3d &mousePos,
                         const Eigen::MatrixX3d &orig_tri_centroids,
                         const Eigen::MatrixX3d &orig_quad_centroids,
                         const std::vector<adjacent_face> &adjacentFaces)
{
    for (int i = 0; i < params_.constraintIters; i++)
    {
        if (params_.pinEnabled)
        {
            compute_pin_constraint(Q, origQ, pinnedVerts);
            std::cout << "Project the pin constraints\n";
        }

        if (params_.stretchEnabled)
        {
            compute_stretch_constraint(Q, origQ, F, orig_tri_centroids);
            std::cout << "Project the strech constraints\n";
        }

        if (params_.bendingEnabled)
        {
            compute_bending_constraint(Q, origQ, orig_quad_centroids, adjacentFaces);
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

    @param Q: The position of the vertices
    @param Qdot: The velocity of the vertices
    @param origQ: The original position of the vertices
    @param F: The vertex index that define the faces of the mesh
    @param pinnedVerts: The indices of the pinned vertices
    @param clickedVertex: The index of the clicked vertex by the mouse
    @param mousePos: The position of the dragged mouse
    @param orig_tri_centroids: The original centroids of the triangles
    @param orig_quad_centroids: The original centroids of the quads
    @param adjacentFaces: The adjacent faces of the vertices
*/
void numericalIntegration(Eigen::MatrixXd &Q,
                          Eigen::MatrixXd &Qdot,
                          const Eigen::MatrixXd &origQ,
                          const Eigen::MatrixXi &F,
                          const std::vector<int> &pinnedVerts,
                          const int &clickedVertex,
                          const Eigen::Vector3d &mousePos,
                          const Eigen::MatrixX3d &orig_tri_centroids,
                          const Eigen::MatrixX3d &orig_quad_centroids,
                          const std::vector<adjacent_face> &adjacentFaces)
{
    Eigen::MatrixXd Q_old = Q;
    Q = Q + params_.timeStep * Qdot;

    project_constraints(Q,
                        origQ,
                        F,
                        pinnedVerts,
                        clickedVertex,
                        mousePos,
                        orig_tri_centroids,
                        orig_quad_centroids,
                        adjacentFaces);

    Eigen::MatrixXd force;
    computeForce(Q, force);

    Qdot = (Q - Q_old) / params_.timeStep;
    Qdot = Qdot + params_.timeStep * force;

    std::cout << "Numerically integrate to update the position and velocity\n";
}