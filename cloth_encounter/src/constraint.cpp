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
    The stretching constraint restores a triangle in the current configuration to its original shape.
    @param Q: the current vertex positions
    @param origQ: the original vertex positions
    @param F: the vertex id of the triangle to be restored
*/
void compute_stretch_constraint(Eigen::MatrixXd &Q,
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

    // Linearly interpolate between the original and current position
    Q.row(vert1_id) = params_.stretchWeight * transformed_orig_triangle.row(0) + (1 - params_.stretchWeight) * Q.row(vert1_id);
    Q.row(vert2_id) = params_.stretchWeight * transformed_orig_triangle.row(1) + (1 - params_.stretchWeight) * Q.row(vert2_id);
    Q.row(vert3_id) = params_.stretchWeight * transformed_orig_triangle.row(2) + (1 - params_.stretchWeight) * Q.row(vert3_id);
}

/*
    The bending constraint restores a pair of triangles within a diamond to their original shape in the rest configuration.
*/
// Function to calculate the centroid of a triangle given by the indices of its vertices
Eigen::Vector3d calculate_triangle_centroid(const Eigen::MatrixXd &vertices, int i, int j, int k) {
    Eigen::Vector3d centroid = (vertices.row(i) + vertices.row(j) + vertices.row(k)) / 3.0;
    return centroid;
}

// Function to calculate the centroid of a quadrilateral
Eigen::Vector3d calculate_quadrilateral_centroid(const Eigen::MatrixXd &vertices) {
    // Assume vertices are in order: 0, 1, 2, 3
    // First diagonal divides the quadrilateral into triangles (0,1,2) and (0,2,3)
    Eigen::Vector3d centroid1 = calculate_triangle_centroid(vertices, 0, 1, 2);
    Eigen::Vector3d centroid2 = calculate_triangle_centroid(vertices, 0, 2, 3);

    // Second diagonal divides the quadrilateral into triangles (0,1,3) and (1,2,3)
    Eigen::Vector3d centroid3 = calculate_triangle_centroid(vertices, 0, 1, 3);
    Eigen::Vector3d centroid4 = calculate_triangle_centroid(vertices, 1, 2, 3);

    // Lines connecting centroids: (centroid1, centroid2) and (centroid3, centroid4)
    // For simplicity, calculate the intersection as the average of all four centroids
    Eigen::Vector3d finalCentroid = (centroid1 + centroid2 + centroid3 + centroid4) / 4.0;
    
    return finalCentroid;
}

void compute_transformation_quad(const Eigen::Matrix<double, 4, 3> &quad,
                            const Eigen::Matrix<double, 4, 3> &orig_quad,
                            Eigen::Vector3d &centroid,
                            Eigen::Vector3d &orig_centroid,
                            Eigen::Matrix3d &rotation)
{   
    // for (uint i=0; i<quad.rows(); ++i){

    // }
    // Eigen::Matrix<double, 4, 3> A;
    // auto x = quad.row(0) - centroid;
    Eigen::Matrix<double, 3, 4> A = (quad.rowwise() - centroid.transpose()).transpose();
    Eigen::Matrix<double, 3, 4> B = (orig_quad.rowwise() - orig_centroid.transpose()).transpose();
    Eigen::Matrix3d AB = A * B.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(AB, Eigen::ComputeFullU | Eigen::ComputeFullV);
    rotation = svd.matrixU() * svd.matrixV().transpose();
}

void compute_bending_constraint(Eigen::MatrixXd &Q, const Eigen::MatrixXd &origQ,
                                const Eigen::MatrixXi &F)
{
    // get pairs of faces that have a common edge  
    // F0 v1, v2, v3
    // F1 v1, v2, v4
    // if any 2 faces more than 2 vertices in common then they qualify for D
      std::vector<std::pair<int, int>> adjacentFaces;
    // Find pairs of faces that share an edge
    for (int i = 0; i < F.rows(); ++i) {
        for (int j = i + 1; j < F.rows(); ++j) {
            std::vector<int> commonVertices;
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    if (F(i, k) == F(j, l)) {
                        commonVertices.push_back(F(i, k));
                    }
                }
            }
            if (commonVertices.size() == 2) {  // Two common vertices mean a shared edge
                adjacentFaces.push_back(std::make_pair(i, j));
            }
        }
    }

    // Process each quadrilateral formed by adjacent faces
    for (auto &pair : adjacentFaces) {
        Eigen::Vector3i F0 = F.row(pair.first);
        Eigen::Vector3i F1 = F.row(pair.second);
        
        // Determine the unique vertex in F1 not in F0
        std::vector<int> uniqueVerticesF1;
        for (int l = 0; l < 3; ++l) {
            if (F0(0) != F1(l) && F0(1) != F1(l) && F0(2) != F1(l)) {
                uniqueVerticesF1.push_back(F1(l));
            }
        }

        // Construct the quadrilateral using vertices
        Eigen::Matrix<double, 4, 3> quad, orig_quad;
        quad.row(0) = Q.row(F0(0));
        quad.row(1) = Q.row(F0(1));
        quad.row(2) = Q.row(F0(2));
        quad.row(3) = Q.row(uniqueVerticesF1[0]); // Non-shared vertex
        
        orig_quad.row(0) = origQ.row(F0(0));
        orig_quad.row(1) = origQ.row(F0(1));
        orig_quad.row(2) = origQ.row(F0(2));
        orig_quad.row(3) = origQ.row(uniqueVerticesF1[0]); // Same for original positions
        Eigen::Vector3d centroid = calculate_quadrilateral_centroid(quad);
        Eigen::Vector3d orig_centroid = calculate_quadrilateral_centroid(orig_quad);

        Eigen::Matrix3d rotation;
        compute_transformation_quad(quad, orig_quad, centroid, orig_centroid, rotation);

        // Center the original quad at the origin
        // Then rotate and translate it back to the centroid of the current quad
        Eigen::Matrix<double, 4, 3> centered_orig_quad = orig_quad.rowwise() - orig_centroid.transpose();
        Eigen::Matrix<double, 4, 3> transformed_orig_quad = ((rotation * centered_orig_quad.transpose()).colwise() + centroid).transpose();

        // // Linearly interpolate between the original and current position
        Q.row(F0(0)) = params_.stretchWeight * transformed_orig_quad.row(0) + (1 - params_.stretchWeight) * Q.row(F0(0));
        Q.row(F0(1)) = params_.stretchWeight * transformed_orig_quad.row(1) + (1 - params_.stretchWeight) * Q.row(F0(1));
        Q.row(F0(2)) = params_.stretchWeight * transformed_orig_quad.row(2) + (1 - params_.stretchWeight) * Q.row(F0(2));
        Q.row(uniqueVerticesF1[0]) = params_.stretchWeight * transformed_orig_quad.row(2) + (1 - params_.stretchWeight) * Q.row(uniqueVerticesF1[0]);
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