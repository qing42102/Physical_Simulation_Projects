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
void compute_bending_constraint(Eigen::MatrixXd &Q, const Eigen::MatrixXd &origQ,
                                const Eigen::MatrixXi &F)
{
    // get pairs of faces that have a common edge  
    // F0 v1, v2, v3
    // F1 v1, v2, v4
    // if any 2 faces more than 2 vertices in common then they qualify for D
    for(uint i=0; i< F.rows(); ++i){
        Eigen::Vector3i F0 = F.row(i);
        for(uint j=i+1; j< F.rows(); ++j){
            Eigen::Vector3i F1 = F.row(j);
            // check if both have more than 2 vertices in common
            int vcnt = 0;
            for(uint k=0; k<3; ++k){
                if (F0(0) == F1(k) || F0(1) == F1(k) || F(2) == F1(k)){
                    ++vcnt;
                }
            }
            if(vcnt == 2){
                compute_stretch_constraint(Q, origQ, F0);
                compute_stretch_constraint(Q, origQ, F1);
            }
                
        }
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