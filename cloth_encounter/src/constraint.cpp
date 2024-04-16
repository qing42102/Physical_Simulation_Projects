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
                            Eigen::Vector3d &translation,
                            Eigen::Matrix3d &rotation)
{
    Eigen::Vector3d centroid = triangle.colwise().mean();
    Eigen::Vector3d orig_centroid = orig_triangle.colwise().mean();
    translation = centroid - orig_centroid;

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
    The stretching constraint restores a triangle to its original shape in the rest configuration.
*/
Eigen::MatrixXd compute_stretch_constraint(const Eigen::MatrixXd &Q,
                                           const Eigen::MatrixXd &origQ,
                                           const Eigen::Vector3i &F)
{
    Eigen::MatrixXd Q_proj = Q;

    int vert1_id = F(0);
    int vert2_id = F(1);
    int vert3_id = F(2);

    Eigen::Matrix3d triangle;
    triangle << Q.row(vert1_id),
        Q.row(vert2_id),
        Q.row(vert3_id);

    Eigen::Matrix3d orig_triangle;
    orig_triangle << origQ.row(vert1_id),
        origQ.row(vert2_id),
        origQ.row(vert3_id);

    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    compute_transformation(triangle, orig_triangle, translation, rotation);

    if (vert1_id == 2025 || vert2_id == 2025 || vert3_id == 2025)
    {
        std::cout << rotation << std::endl;
        std::cout << translation << std::endl;
    }

    // Eigen::Matrix3d transformed_triangle = ((rotation * orig_triangle.transpose()).colwise() + translation).transpose();
    // Q_proj(Eigen::placeholders::all, {vert1_id, vert2_id, vert3_id}) = transformed_triangle;
    Q_proj.row(vert1_id) = (rotation * origQ.row(vert1_id).transpose() + translation).transpose();
    Q_proj.row(vert2_id) = (rotation * origQ.row(vert2_id).transpose() + translation).transpose();
    Q_proj.row(vert3_id) = (rotation * origQ.row(vert3_id).transpose() + translation).transpose();

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
Eigen::MatrixXd compute_pull_constraint(const Eigen::MatrixXd &Q)
{
    Eigen::MatrixXd Q_proj = Q;
    return Q_proj;
}