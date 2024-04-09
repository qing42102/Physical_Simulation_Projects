#include <vector>
#include <unordered_map>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/signed_distance.h>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"

/*
    This struct stores for a vertex: the signed distance from the vertex to the surface of the colliding body,
    the derivative of the signed distance, and the index of the colliding body.
*/
struct collision_point_data
{
    int collidied_body;
    double signed_dist;
    double signed_dist_deriv;

    collision_point_data() {}
    collision_point_data(int collidied_body, double signed_dist, double signed_dist_deriv)
    {
        collidied_body = collidied_body;
        signed_dist = signed_dist;
        signed_dist_deriv = signed_dist_deriv;
    }
};

/*
    This struct stores all the collision data for a single body.
    The collision data is stored as a hash map from vertex index to a vector of collision points.
*/
struct collision_data
{
    int body;
    std::unordered_map<int, std::vector<collision_point_data>> collisions;

    collision_data() {}
    collision_data(int body) : body(body) {}

    void add_collision(const int &vert,
                       const int &collidied_body,
                       const double &signed_dist,
                       const double &signed_dist_deriv)
    {
        collisions[vert].push_back(collision_point_data(collidied_body, signed_dist, signed_dist_deriv));
    }

    std::vector<collision_point_data> get_collisions(const int &vert)
    {
        return collisions[vert];
    }
};

/*
    This function computes the bounding box of a rigid body.
    Construct a cubic box around the body with side length 2 * radius
    The bounding box is a 3x8 matrix where each column is a vertex of the box.
*/
Eigen::MatrixXd bounding_box(const RigidBodyInstance *body)
{
    double radius = body->getTemplate().getBoundingRadius();
    Eigen::Vector3d center = body->c;
    Eigen::Matrix3d rotation = VectorMath::rotationMatrix(body->theta);

    Eigen::MatrixXd box(3, 8);
    box.col(0) = Eigen::Vector3d(-radius, -radius, -radius);
    box.col(1) = Eigen::Vector3d(-radius, -radius, radius);
    box.col(2) = Eigen::Vector3d(-radius, radius, -radius);
    box.col(3) = Eigen::Vector3d(-radius, radius, radius);
    box.col(4) = Eigen::Vector3d(radius, -radius, -radius);
    box.col(5) = Eigen::Vector3d(radius, -radius, radius);
    box.col(6) = Eigen::Vector3d(radius, radius, -radius);
    box.col(7) = Eigen::Vector3d(radius, radius, radius);

    // Rotate and translate the box to the current configuration
    box = rotation * box;
    box.colwise() += center;

    return box;
}

bool bounding_box_overlap(const Eigen::MatrixXd &box1, const Eigen::MatrixXd &box2)
{
    for (int i = 0; i < 8; i++)
    {
        // Check if any of the vertices of box2 are inside box1
        // Checking if the x, y, and z coordinates of the vertex are within the range of box1
        if (box1.col(0).minCoeff() <= box2.col(i).maxCoeff() && box1.col(0).maxCoeff() >= box2.col(i).minCoeff() &&
            box1.col(1).minCoeff() <= box2.col(i).maxCoeff() && box1.col(1).maxCoeff() >= box2.col(i).minCoeff() &&
            box1.col(2).minCoeff() <= box2.col(i).maxCoeff() && box1.col(2).maxCoeff() >= box2.col(i).minCoeff())
        {
            return true;
        }
    }
    return false;
}

double signed_distance_deriv()
{
    return 0;
}

/*
    This function performs narrow-phase collision detection between two rigid bodies.
    Use the signed distance function to determine if any of the vertices of the first body are inside the second body.
    Adds all the vertices are in collision to the collision_data struct.
*/
void vertex_body_overlap(const RigidBodyInstance *body1,
                         const RigidBodyInstance *body2,
                         const int &body2_num,
                         collision_data &body_collision)
{
    // First rotate and translate the vertices of body1 to the current configuration
    Eigen::Vector3d center1 = body1->c;
    Eigen::Matrix3d rotation1 = VectorMath::rotationMatrix(body1->theta);
    Eigen::MatrixX3d verts = body1->getTemplate().getVerts();
    Eigen::MatrixX3d verts_current_config = ((rotation1 * verts.transpose()).colwise() + center1).transpose();

    // Then rotate and translate the vertices to the template configuration of body2
    Eigen::Vector3d center2 = body2->c;
    Eigen::Matrix3d rotation2 = VectorMath::rotationMatrix(body2->theta);
    Eigen::MatrixX3d verts_body2_config = (rotation2.transpose() * (verts_current_config.transpose().colwise() - center2)).transpose();

    // N is the normal vector of the closest point on the surface of body2 to each vertex of body1
    // S is the signed distance from each vertex of body1 to the surface of body2
    // I is the index of the closest point on the surface of body2 to each vertex of body1
    // C is the closest point on the surface of body2 to each vertex of body1
    Eigen::MatrixX3d V = body2->getTemplate().getVerts();
    Eigen::MatrixX3i F = body2->getTemplate().getFaces();
    Eigen::MatrixX3d N;
    Eigen::VectorXd S;
    Eigen::VectorXi I;
    Eigen::MatrixX3d C;
    igl::SignedDistanceType sign_type = igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
    igl::signed_distance(verts_body2_config, V, F, sign_type, S, I, C, N);

    // If the signed distance is negative, the vertex is inside the body
    for (int i = 0; i < verts_current_config.rows(); i++)
    {
        if (S(i) < 0)
        {
            double signed_dist_deriv = signed_distance_deriv();

            body_collision.add_collision(i, body2_num, S(i), signed_dist_deriv);
        }
    }
}

std::vector<collision_data> detect_collisions()
{
    std::vector<Eigen::MatrixXd> boxes(bodies_.size());
    for (uint i = 0; i < bodies_.size(); i++)
    {
        Eigen::MatrixXd box = bounding_box(bodies_[i]);
        boxes[i] = box;
    }

    std::vector<collision_data> all_collisions(bodies_.size());
    for (uint i = 0; i < bodies_.size(); i++)
    {
        collision_data body_collision(i);
        for (uint j = i + 1; j < bodies_.size(); j++)
        {
            // Broad-phase collision detection using bounding boxes
            if (bounding_box_overlap(boxes[i], boxes[j]))
            {
                // Perform narrow-phase collision detection through vertex-body overlap
                vertex_body_overlap(bodies_[i], bodies_[j], j, body_collision);
            }
        }

        all_collisions[i] = body_collision;
    }

    return all_collisions;
}