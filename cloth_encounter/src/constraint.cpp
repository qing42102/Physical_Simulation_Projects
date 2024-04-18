#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"

/*
    Compute the optimal translation and rotation from the original triangle to the current stretched triangle
    @param triangle: the current triangle where each row is a vertex
    @param orig_triangle: the original triangle where each row is a vertex
*/
void compute_transformation(const Eigen::Matrix3d &triangle,
                            const Eigen::Matrix3d &orig_triangle,
                            Eigen::Vector3d &centroid,
                            Eigen::Vector3d &orig_centroid,
                            Eigen::Matrix3d &rotation)
{
    centroid = triangle.colwise().mean();
    orig_centroid = orig_triangle.colwise().mean();

    Eigen::Matrix3d A = (triangle.rowwise() - centroid.transpose()).transpose();
    Eigen::Matrix3d B = (orig_triangle.rowwise() - orig_centroid.transpose()).transpose();
    Eigen::Matrix3d AB = A * B.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(AB, Eigen::ComputeFullU | Eigen::ComputeFullV);
    rotation = svd.matrixU() * svd.matrixV().transpose();
}

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
    The stretching constraint restores a triangle in the current configuration to its original shape.
    @param Q: the current vertex positions
    @param origQ: the original vertex positions
    @param F: the vertex id of the triangle to be restored
*/
Eigen::MatrixXd compute_stretch_constraint(const Eigen::MatrixXd &Q,
                                           const Eigen::MatrixXd &origQ,
                                           const Eigen::Vector3i &F)
{
    int vert1_id = F(0);
    int vert2_id = F(1);
    int vert3_id = F(2);

    Eigen::Matrix3d triangle;
    triangle.row(0) = Q.row(vert1_id);
    triangle.row(1) = Q.row(vert2_id);
    triangle.row(2) = Q.row(vert3_id);

    Eigen::Matrix3d orig_triangle;
    orig_triangle.row(0) = origQ.row(vert1_id);
    orig_triangle.row(1) = origQ.row(vert2_id);
    orig_triangle.row(2) = origQ.row(vert3_id);

    Eigen::Vector3d centroid;
    Eigen::Vector3d orig_centroid;
    Eigen::Matrix3d rotation;
    compute_transformation(triangle, orig_triangle, centroid, orig_centroid, rotation);

    // Center the original triangle at the origin
    // Then rotate and translate it back to the centroid of the current triangle
    Eigen::Matrix3d centered_orig_triangle = orig_triangle.rowwise() - orig_centroid.transpose();
    Eigen::Matrix3d transformed_orig_triangle = ((rotation * centered_orig_triangle.transpose()).colwise() + centroid).transpose();

    Eigen::Matrix3d Q_proj = transformed_orig_triangle;

    return Q_proj;
}

/*
    The bending constraint restores a pair of triangles within a diamond to their original shape in the rest configuration.
*/
Eigen::MatrixXd compute_bending_constraint(const Eigen::MatrixXd &Q)
{
    Eigen::MatrixXd Q_proj = Q;
    return Q_proj;
}

/*
    Constraint for placing the dragged vertex at the current location of the mouse pointer.
*/
Eigen::MatrixXd compute_pull_constraint(const Eigen::MatrixXd &Q,
                                        const int &clickedVertex,
                                        const Eigen::Vector3d &mousePos)
{
    Eigen::MatrixXd Q_proj = Q;
    Q_proj.row(clickedVertex) = mousePos;
    return Q_proj;
}