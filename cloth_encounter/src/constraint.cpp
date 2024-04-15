#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"

/*
    Constraint for stopping the top-left and top-right corners of the cloth from moving.
*/
Eigen::MatrixXd compute_pin_constraint(const Eigen::MatrixXd &Q,
                                       const Eigen::MatrixXd &origQ,
                                       const std::vector<int> &pinnedVerts)
{
    Eigen::MatrixXd Q_proj = Q;
    for (uint i = 0; i < pinnedVerts.size(); i++)
    {
        // Prevent the pinned vertex from moving by setting it to the original position
        int vert_id = pinnedVerts[i];
        Q_proj.row(vert_id) = origQ.row(vert_id);
    }

    return Q_proj;
}

/*
    The stretching constraint restores a triangle to its original shape in the rest configuration.
*/
Eigen::MatrixXd compute_stretch_constraint(const Eigen::MatrixXd &Q)
{
    Eigen::MatrixXd Q_proj;
    return Q_proj;
}

/*
    The bending constraint restores a pair of triangles within a diamond to their original shape in the rest configuration.
*/
Eigen::MatrixXd compute_bending_constraint(const Eigen::MatrixXd &Q)
{
    Eigen::MatrixXd Q_proj;
    return Q_proj;
}

/*
    Constraint for placing the dragged vertex at the current location of the mouse pointer.
*/
Eigen::MatrixXd compute_pull_constraint(const Eigen::MatrixXd &Q)
{
    Eigen::MatrixXd Q_proj;
    return Q_proj;
}