#include <iostream>
#include <set>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#pragma once
#include "SimParameters.h"

/*
    Struct to store the adjacent faces that share an edge
    Stores the face indices, the unique vertices, and the shared vertices
*/
struct adjacent_face
{
    int face1;
    int face2;

    int unique_vert1;
    int unique_vert2;

    int shared_vert1;
    int shared_vert2;

    adjacent_face() {}
    adjacent_face(int f1, int f2, int uv1, int uv2, int sv1, int sv2)
    {
        face1 = f1;
        face2 = f2;
        unique_vert1 = uv1;
        unique_vert2 = uv2;
        shared_vert1 = sv1;
        shared_vert2 = sv2;
    }
};

/*
    Constraint for stopping the top-left and top-right corners of the cloth from moving.
*/
void compute_pin_constraint(Eigen::MatrixXd &Q,
                            const Eigen::MatrixXd &origQ,
                            const std::vector<int> &pinnedVerts)
{
    for (uint i = 0; i < pinnedVerts.size(); i++)
    {
        // Prevent the pinned vertex from moving by setting it to the original position
        int vert_id = pinnedVerts[i];
        Eigen::Vector3d fixed_vert = origQ.row(vert_id);

        // Linearly interpolate between the original and current position
        Q.row(vert_id) = params_.pinWeight * fixed_vert.transpose() + (1 - params_.pinWeight) * Q.row(vert_id);
    }
}

/*
    Function to calculate the centroid of a triangle
*/
Eigen::Vector3d calc_tri_centroid(const Eigen::Matrix3d &triangle)
{
    Eigen::Vector3d centroid = triangle.colwise().mean();
    return centroid;
}

/*
    Compute the optimal translation and rotation from the original shape to the current stretched shape
    Shape can be a triangle or a quadrilateral
    Solves the orthogonal Procrustes problem
    @param shape: the current vertex positions of size 3x3 or 4x3
    @param orig_shape: the original vertex positions of size 3x3 or 4x3
    @param centroid: the centroid of the current shape
    @param orig_centroid: the centroid of the original shape
    @param rotation: the optimal rotation matrix
*/
void compute_shape_transform(const Eigen::MatrixXd &shape,
                             const Eigen::MatrixXd &orig_shape,
                             const Eigen::Vector3d &centroid,
                             const Eigen::Vector3d &orig_centroid,
                             Eigen::Matrix3d &rotation)
{
    Eigen::MatrixXd A = (shape.rowwise() - centroid.transpose()).transpose();
    Eigen::MatrixXd B = (orig_shape.rowwise() - orig_centroid.transpose()).transpose();
    Eigen::Matrix3d AB = A * B.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(AB, Eigen::ComputeFullU | Eigen::ComputeFullV);
    rotation = svd.matrixU() * svd.matrixV().transpose();
}

/*
    The stretching constraint restores a triangle in the current configuration to its original shape.
*/
void compute_stretch_constraint(Eigen::MatrixXd &Q,
                                const Eigen::MatrixXd &origQ,
                                const Eigen::MatrixXi &F,
                                const Eigen::MatrixX3d &orig_tri_centroids)
{
    for (int i = 0; i < F.rows(); i++)
    {
        int vert1_id = F(i, 0);
        int vert2_id = F(i, 1);
        int vert3_id = F(i, 2);

        Eigen::Matrix3d triangle;
        triangle.row(0) = Q.row(vert1_id);
        triangle.row(1) = Q.row(vert2_id);
        triangle.row(2) = Q.row(vert3_id);

        Eigen::Matrix3d orig_triangle;
        orig_triangle.row(0) = origQ.row(vert1_id);
        orig_triangle.row(1) = origQ.row(vert2_id);
        orig_triangle.row(2) = origQ.row(vert3_id);

        Eigen::Vector3d centroid = calc_tri_centroid(triangle);
        Eigen::Vector3d orig_centroid = orig_tri_centroids.row(i);
        Eigen::Matrix3d rotation;
        compute_shape_transform(triangle, orig_triangle, centroid, orig_centroid, rotation);

        // Center the original triangle at the origin
        // Then rotate and translate it back to the centroid of the current triangle
        Eigen::Matrix3d centered_orig_triangle = orig_triangle.rowwise() - orig_centroid.transpose();
        Eigen::Matrix3d transformed_orig_triangle = ((rotation * centered_orig_triangle.transpose()).colwise() + centroid).transpose();

        // Linearly interpolate between the original and current position
        Q.row(vert1_id) = params_.stretchWeight * transformed_orig_triangle.row(0) + (1 - params_.stretchWeight) * Q.row(vert1_id);
        Q.row(vert2_id) = params_.stretchWeight * transformed_orig_triangle.row(1) + (1 - params_.stretchWeight) * Q.row(vert2_id);
        Q.row(vert3_id) = params_.stretchWeight * transformed_orig_triangle.row(2) + (1 - params_.stretchWeight) * Q.row(vert3_id);
    }
}

/*
    Function to calculate the centroid of a quadrilateral
*/
Eigen::Vector3d calc_quad_centroid(const Eigen::Matrix<double, 4, 3> &quad)
{
    // Assume vertices are in order: 0, 1, 2, 3
    // First diagonal divides the quadrilateral into triangles (0,1,2) and (0,2,3)
    Eigen::Vector3d centroid1 = (quad.row(0) + quad.row(1) + quad.row(2)) / 3.0;
    Eigen::Vector3d centroid2 = (quad.row(0) + quad.row(2) + quad.row(3)) / 3.0;

    // Second diagonal divides the quadrilateral into triangles (0,1,3) and (1,2,3)
    Eigen::Vector3d centroid3 = (quad.row(0) + quad.row(1) + quad.row(3)) / 3.0;
    Eigen::Vector3d centroid4 = (quad.row(1) + quad.row(2) + quad.row(3)) / 3.0;

    // Lines connecting centroids: (centroid1, centroid2) and (centroid3, centroid4)
    // For simplicity, calculate the intersection as the average of all four centroids
    Eigen::Vector3d finalCentroid = (centroid1 + centroid2 + centroid3 + centroid4) / 4.0;

    return finalCentroid;
}

/*
    Find the adjacent faces that share an edge
    Check if two faces share 2 vertices
*/
std::vector<adjacent_face> compute_adjacent_faces(const Eigen::MatrixXi &F)
{
    // Find pairs of faces that share an edge
    // If faces share 2 vertices, then they share an edge and form a quadrilateral
    std::vector<adjacent_face> adjacentFaces;
    for (int i = 0; i < F.rows(); ++i)
    {
        Eigen::Vector3i F0 = F.row(i);
        std::set<int> F0_verts = {F0(0), F0(1), F0(2)};

        // Check pairwise with all other faces
        for (int j = i + 1; j < F.rows(); ++j)
        {
            Eigen::Vector3i F1 = F.row(j);
            std::set<int> F1_verts = {F1(0), F1(1), F1(2)};

            // The unique verts are the set difference between the vertices of the two faces
            std::vector<int> unique_verts;
            std::set_symmetric_difference(F0_verts.begin(),
                                          F0_verts.end(),
                                          F1_verts.begin(),
                                          F1_verts.end(),
                                          std::inserter(unique_verts, unique_verts.begin()));

            // The common verts are the intersection of the vertices of the two faces
            std::vector<int> common_verts;
            std::set_intersection(F0_verts.begin(),
                                  F0_verts.end(),
                                  F1_verts.begin(),
                                  F1_verts.end(),
                                  std::inserter(common_verts, common_verts.begin()));

            // Two common vertices mean a shared edge
            if (common_verts.size() == 2)
            {
                adjacent_face af(i, j, unique_verts[0], unique_verts[1], common_verts[0], common_verts[1]);
                adjacentFaces.push_back(af);
            }
        }
    }

    return adjacentFaces;
}

/*
    The bending constraint restores a pair of triangles within a diamond to their original shape in the rest configuration.
*/
void compute_bending_constraint(Eigen::MatrixXd &Q,
                                const Eigen::MatrixXd &origQ,
                                const Eigen::MatrixX3d &orig_quad_centroids,
                                const std::vector<adjacent_face> &adjacentFaces)
{
    // Process each quadrilateral formed by adjacent faces
    for (uint i = 0; i < adjacentFaces.size(); i++)
    {
        // Construct the quadrilateral using a specific order of the shared and unique vertices
        Eigen::Matrix<double, 4, 3> quad, orig_quad;
        quad.row(0) = Q.row(adjacentFaces[i].shared_vert1);
        quad.row(1) = Q.row(adjacentFaces[i].unique_vert1);
        quad.row(2) = Q.row(adjacentFaces[i].shared_vert2);
        quad.row(3) = Q.row(adjacentFaces[i].unique_vert2);

        orig_quad.row(0) = origQ.row(adjacentFaces[i].shared_vert1);
        orig_quad.row(1) = origQ.row(adjacentFaces[i].unique_vert1);
        orig_quad.row(2) = origQ.row(adjacentFaces[i].shared_vert2);
        orig_quad.row(3) = origQ.row(adjacentFaces[i].unique_vert2);

        Eigen::Vector3d centroid = calc_quad_centroid(quad);
        Eigen::Vector3d orig_centroid = orig_quad_centroids.row(i);

        Eigen::Matrix3d rotation;
        compute_shape_transform(quad, orig_quad, centroid, orig_centroid, rotation);

        // Center the original quad at the origin
        // Then rotate and translate it back to the centroid of the current quad
        Eigen::Matrix<double, 4, 3> centered_orig_quad = orig_quad.rowwise() - orig_centroid.transpose();
        Eigen::Matrix<double, 4, 3> transformed_orig_quad = ((rotation * centered_orig_quad.transpose()).colwise() + centroid).transpose();

        // Linearly interpolate between the original and current position
        Q.row(adjacentFaces[i].shared_vert1) = params_.bendingWeight * transformed_orig_quad.row(0) + (1 - params_.bendingWeight) * Q.row(adjacentFaces[i].shared_vert1);
        Q.row(adjacentFaces[i].unique_vert1) = params_.bendingWeight * transformed_orig_quad.row(1) + (1 - params_.bendingWeight) * Q.row(adjacentFaces[i].unique_vert1);
        Q.row(adjacentFaces[i].shared_vert2) = params_.bendingWeight * transformed_orig_quad.row(2) + (1 - params_.bendingWeight) * Q.row(adjacentFaces[i].shared_vert2);
        Q.row(adjacentFaces[i].unique_vert2) = params_.bendingWeight * transformed_orig_quad.row(3) + (1 - params_.bendingWeight) * Q.row(adjacentFaces[i].unique_vert2);
    }
}

/*
    Constraint for placing the dragged vertex at the current location of the mouse pointer.
*/
void compute_pull_constraint(Eigen::MatrixXd &Q,
                             const int &clickedVertex,
                             const Eigen::Vector3d &mousePos)
{
    // Linearly interpolate between the original and current position
    Q.row(clickedVertex) = params_.pullingWeight * mousePos.transpose() + (1 - params_.pullingWeight) * Q.row(clickedVertex);
}