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
    This struct stores for a vertex:
    - collidied_body: the index of the colliding body
    - signed_dist: the signed distance from the vertex to the surface of the colliding body
    - signed_dist_deriv: the derivative of the signed distance
*/
struct collision_point_data
{
    uint collidied_body = -1;
    double signed_dist;
    Eigen::Vector3d signed_dist_deriv;

    collision_point_data() {}
    collision_point_data(uint collidied_body_, double signed_dist_, Eigen::Vector3d signed_dist_deriv_)
    {
        collidied_body = collidied_body_;
        signed_dist = signed_dist_;
        signed_dist_deriv = signed_dist_deriv_;
    }
};

/*
    This struct stores all the collision data for a single body.
    The collision data is stored as a hash map from vertex index to a vector of collisions with another body.
*/
struct collision_data
{
    int body;
    std::unordered_map<int, std::vector<collision_point_data>> collisions;

    collision_data() {}
    collision_data(int body_) : body(body_) {}

    void add_collision(const uint &vert,
                       const uint &collidied_body,
                       const double &signed_dist,
                       const Eigen::Vector3d &signed_dist_deriv)
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
Eigen::MatrixXd bounding_box(const Eigen::Vector3d &trans_pos,
                             const Eigen::Vector3d &angle,
                             const RigidBodyInstance *body)
{
    double radius = body->getTemplate().getBoundingRadius();
    Eigen::Matrix3d rotation = VectorMath::rotationMatrix(angle);

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
    box.colwise() += trans_pos;

    return box;
}

bool bounding_box_overlap(const Eigen::MatrixXd &box1, const Eigen::MatrixXd &box2)
{
    // Check if any of the vertices of box2 are inside box1
    // Checking if the x, y, and z coordinates of the vertex are within the range of box1
    if (box1.row(0).minCoeff() <= box2.row(0).maxCoeff() && box1.row(0).maxCoeff() >= box2.row(0).minCoeff() &&
        box1.row(1).minCoeff() <= box2.row(1).maxCoeff() && box1.row(1).maxCoeff() >= box2.row(1).minCoeff() &&
        box1.row(2).minCoeff() <= box2.row(2).maxCoeff() && box1.row(2).maxCoeff() >= box2.row(2).minCoeff())
    {
        return true;
    }

    return false;
}

/*
    This function computes the derivative of the signed distance function.
    The derivative is the normalized vector from the vertex to the closest point on the surface of the colliding body.
    @param P: The vertices of the body
    @param C: The closest point on the surface of the colliding body to each vertex
    @param S: The signed distance from each vertex to the surface of the colliding body
*/
Eigen::MatrixX3d signed_distance_deriv(Eigen::MatrixX3d P, Eigen::MatrixX3d C, Eigen::VectorXd S)
{
    assert(P.rows() == S.size());

    // Normalize each row by the squared norm of each row
    Eigen::VectorXd norm = (P - C).rowwise().squaredNorm();
    Eigen::MatrixX3d dir = (P - C).array().colwise() / norm.array();

    Eigen::MatrixX3d S_deriv = dir.array().colwise() * S.array();

    return S_deriv;
}

/*
    This function performs narrow-phase collision detection between two rigid bodies.
    Use the signed distance function to determine if any of the vertices of the first body are inside the second body.
    Adds all the vertices that are in collision to the collision_data struct.
*/
void vertex_body_overlap(const Eigen::VectorXd &trans_pos,
                         const Eigen::VectorXd &angle,
                         const RigidBodyInstance *body1,
                         const RigidBodyInstance *body2,
                         const uint &body1_num,
                         const uint &body2_num,
                         collision_data &body_collision)
{
    // First rotate and translate the vertices of body1 to the current configuration
    Eigen::Vector3d center1 = trans_pos.segment(3 * body1_num, 3);
    Eigen::Matrix3d rotation1 = VectorMath::rotationMatrix(angle.segment(3 * body1_num, 3));
    Eigen::MatrixX3d verts = body1->getTemplate().getVerts();
    Eigen::MatrixX3d verts_current_config = ((rotation1 * verts.transpose()).colwise() + center1).transpose();

    // Then rotate and translate the vertices to the template configuration of body2
    Eigen::Vector3d center2 = trans_pos.segment(3 * body2_num, 3);
    Eigen::Matrix3d rotation2 = VectorMath::rotationMatrix(angle.segment(3 * body2_num, 3));
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

    Eigen::MatrixX3d signed_dist_deriv = signed_distance_deriv(verts_body2_config, C, S);
    assert(signed_dist_deriv.rows() == S.size());

    // If the signed distance is negative, the vertex is inside the body
    for (int i = 0; i < verts_current_config.rows(); i++)
    {
        if (S(i) < 0)
        {
            body_collision.add_collision(i, body2_num, S(i), signed_dist_deriv.row(i));
        }
    }
}

/*
    Detect collisions between all pairs of rigid bodies.
    Returns a vector of collision_data structs for each body, where index i corresponds to body i.
*/
std::vector<collision_data> detect_collisions(const Eigen::VectorXd &trans_pos,
                                              const Eigen::VectorXd &angle)
{
    std::vector<Eigen::MatrixXd> boxes(bodies_.size());
    for (uint i = 0; i < bodies_.size(); i++)
    {
        Eigen::MatrixXd box = bounding_box(trans_pos.segment(3 * i, 3),
                                           angle.segment(3 * i, 3),
                                           bodies_[i]);
        boxes[i] = box;
    }

    std::vector<collision_data> all_collisions(bodies_.size());
    for (uint i = 0; i < bodies_.size(); i++)
    {
        collision_data body_collision(i);
        for (uint j = 0; j < bodies_.size(); j++)
        {
            // Broad-phase collision detection using bounding boxes
            if (i != j && bounding_box_overlap(boxes[i], boxes[j]))
            {
                // Perform narrow-phase collision detection through vertex-body overlap
                vertex_body_overlap(trans_pos, angle, bodies_[i], bodies_[j], i, j, body_collision);
            }
        }

        all_collisions[i] = body_collision;
    }

    return all_collisions;
}

/*
    Constrain the vertex to lie above a ground plane at y = −1:
    g(q) = 1 + p \hat y
*/
Eigen::VectorXd floor_constraint(const Eigen::Vector3d &trans_pos,
                                 const Eigen::Vector3d &angle,
                                 const RigidBodyInstance *body)
{
    Eigen::Matrix3d rotation = VectorMath::rotationMatrix(angle);
    Eigen::MatrixX3d verts = body->getTemplate().getVerts();
    Eigen::MatrixX3d verts_current_config = ((rotation * verts.transpose()).colwise() + trans_pos).transpose();

    Eigen::VectorXd constraint = 1 + verts_current_config.col(1).array();

    return constraint;
}

/*
    Derivative of the floor constraint with respect to time, theta, and c
*/
void floor_constraint_deriv(const Eigen::Vector3d &trans_vel,
                            const Eigen::Vector3d &angle,
                            const Eigen::Vector3d &angle_vel,
                            const RigidBodyInstance *body,
                            Eigen::VectorXd &constraint_deriv_t,
                            Eigen::MatrixX3d &constraint_deriv_theta,
                            Eigen::MatrixX3d &constraint_deriv_c)
{
    Eigen::MatrixX3d verts = body->getTemplate().getVerts();

    // Translation derivative is 1 in the y direction
    constraint_deriv_c = Eigen::MatrixX3d::Zero(verts.rows(), 3);
    constraint_deriv_c.col(1).array() = 1;

    constraint_deriv_theta = Eigen::MatrixX3d::Zero(verts.rows(), 3);
    for (int i = 0; i < verts.rows(); i++)
    {
        Eigen::Vector3d vert = verts.row(i).transpose();
        Eigen::Matrix3d deriv = VectorMath::DrotVector(angle, vert);

        // Only y component derivative
        constraint_deriv_theta.row(i) = deriv.row(1);
    }

    // Only y component derivative
    constraint_deriv_t = (VectorMath::DrotVector(angle, angle_vel) * verts.transpose()).transpose().col(1);
    constraint_deriv_t.array() += trans_vel(1);
}

/*
    Compute the constraint that the vertex does not collide with another body:
    Signed distance field s(\theta_i, theta_j, c_i, c_j)
*/
Eigen::VectorXd collision_constraint(const std::vector<collision_point_data> &vert_collisions)
{
    Eigen::VectorXd constraints(vert_collisions.size());

    // Collect all the signed distance collision points for a single vertex
    for (uint i = 0; i < vert_collisions.size(); i++)
    {
        constraints[i] = vert_collisions[i].signed_dist;
    }

    return constraints;
}

/*
    The derivative of the collision constraint with respect to time, theta, and c
*/
void collision_constraint_deriv(const Eigen::VectorXd &trans_pos,
                                const Eigen::VectorXd &trans_vel,
                                const Eigen::VectorXd &angle,
                                const Eigen::VectorXd &angle_vel,
                                const uint &body_i_id,
                                const uint &vertex_i_id,
                                const std::vector<collision_point_data> &vert_collisions,
                                Eigen::VectorXd &constraint_deriv_t,
                                Eigen::MatrixX3d &constraint_deriv_theta_i,
                                Eigen::MatrixX3d &constraint_deriv_c_i,
                                Eigen::MatrixX3d &constraint_deriv_theta_j,
                                Eigen::MatrixX3d &constraint_deriv_c_j)
{
    Eigen::MatrixX3d verts = bodies_[body_i_id]->getTemplate().getVerts();
    Eigen::Vector3d vert = verts.row(vertex_i_id).transpose();

    Eigen::Vector3d trans_vel_i = trans_vel.segment(3 * body_i_id, 3);
    Eigen::Vector3d angle_i = angle.segment(3 * body_i_id, 3);
    Eigen::Vector3d angle_vel_i = angle_vel.segment(3 * body_i_id, 3);
    Eigen::Vector3d trans_pos_i = trans_pos.segment(3 * body_i_id, 3);

    Eigen::Matrix3d rotation_i = VectorMath::rotationMatrix(angle_i);

    // Precompute some terms
    Eigen::Vector3d term1 = VectorMath::DrotVector(angle_i, angle_vel_i) * vert;
    Eigen::Matrix3d term2 = VectorMath::DrotVector(angle_i, vert);
    Eigen::Vector3d term3 = rotation_i * vert + trans_pos_i;

    for (uint k = 0; k < vert_collisions.size(); k++)
    {
        uint body_j_id = vert_collisions[k].collidied_body;
        Eigen::Vector3d signed_dist_deriv = vert_collisions[k].signed_dist_deriv;

        Eigen::Vector3d trans_vel_j = trans_vel.segment(3 * body_j_id, 3);
        Eigen::Vector3d angle_j = angle.segment(3 * body_j_id, 3);
        Eigen::Vector3d angle_vel_j = angle_vel.segment(3 * body_j_id, 3);
        Eigen::Vector3d trans_pos_j = trans_pos.segment(3 * body_j_id, 3);

        Eigen::Matrix3d rotation_j = VectorMath::rotationMatrix(angle_j);

        Eigen::Vector3d term5 = rotation_j.transpose() * term1 + trans_vel_i - trans_vel_j;
        Eigen::Vector3d term6 = VectorMath::DrotVector(-angle_j, -angle_vel_j) * (term3 - trans_pos_j);
        constraint_deriv_t[k] = signed_dist_deriv.transpose() * (term5 + term6);

        constraint_deriv_c_i.row(k) = (signed_dist_deriv.transpose() * rotation_j.transpose());
        constraint_deriv_c_j.row(k) = (signed_dist_deriv.transpose() * -rotation_j.transpose());

        constraint_deriv_theta_i.row(k) = (signed_dist_deriv.transpose() * rotation_j.transpose() * term2);
        constraint_deriv_theta_j.row(k) = (signed_dist_deriv.transpose() * VectorMath::DrotVector(-angle_i, term3 - trans_pos_j));
    }
}

/*
    Add a penalty force if the body violates the floor constraint
*/
void add_penalty_floor(const Eigen::VectorXd &trans_pos,
                       const Eigen::VectorXd &trans_vel,
                       const Eigen::VectorXd &angle,
                       const Eigen::VectorXd &angle_vel,
                       Eigen::VectorXd &F_trans,
                       Eigen::VectorXd &F_angle)
{
    for (uint i = 0; i < bodies_.size(); i++)
    {
        Eigen::VectorXd constraint = floor_constraint(trans_pos.segment(3 * i, 3),
                                                      angle.segment(3 * i, 3),
                                                      bodies_[i]);

        Eigen::VectorXd constraint_deriv_t;
        Eigen::MatrixX3d constraint_deriv_theta;
        Eigen::MatrixX3d constraint_deriv_c;

        floor_constraint_deriv(trans_vel.segment(3 * i, 3),
                               angle.segment(3 * i, 3),
                               angle_vel.segment(3 * i, 3),
                               bodies_[i],
                               constraint_deriv_t,
                               constraint_deriv_theta,
                               constraint_deriv_c);

        assert(constraint.size() == constraint_deriv_t.size());
        assert(constraint.size() == constraint_deriv_theta.rows());
        assert(constraint.size() == constraint_deriv_c.rows());

        // Check whether each vertex violates the constraint
        // Add a penalty force to the rigid body for each vertex that violates the constraint
        for (int j = 0; j < constraint.size(); j++)
        {
            if (constraint(j) < 0 && constraint_deriv_t(j) < 0)
            {
                F_trans.segment(3 * i, 3) += params_.penaltyStiffness * constraint(j) * constraint_deriv_c.row(j).transpose();
                F_angle.segment(3 * i, 3) += params_.penaltyStiffness * constraint(j) * constraint_deriv_theta.row(j).transpose();
            }
            else if (constraint(j) < 0 && constraint_deriv_t(j) >= 0)
            {
                F_trans.segment(3 * i, 3) += params_.penaltyStiffness * params_.coefficientOfRestitution * constraint(j) * constraint_deriv_c.row(j).transpose();
                F_angle.segment(3 * i, 3) += params_.penaltyStiffness * params_.coefficientOfRestitution * constraint(j) * constraint_deriv_theta.row(j).transpose();
            }
        }
    }
}

void add_penalty_collision(
    const std::vector<collision_data> &all_collisions,
    const Eigen::VectorXd &trans_pos,
    const Eigen::VectorXd &trans_vel,
    const Eigen::VectorXd &angle,
    const Eigen::VectorXd &angle_vel,
    Eigen::VectorXd &F_trans,
    Eigen::VectorXd &F_angle)
{
    // i -> body_id
    for (uint i = 0; i < all_collisions.size(); ++i)
    {
        std::unordered_map<int, std::vector<collision_point_data>> collisions = all_collisions[i].collisions;

        // Each vertex of the body
        for (auto it = collisions.begin(); it != collisions.end(); ++it)
        {
            std::vector<collision_point_data> &vert_collisions = it->second;

            int num_collisions = vert_collisions.size();
            Eigen::MatrixX3d constraint_deriv_c_i = Eigen::MatrixX3d::Zero(num_collisions, 3);
            Eigen::MatrixX3d constraint_deriv_theta_i = Eigen::MatrixX3d::Zero(num_collisions, 3);
            Eigen::MatrixX3d constraint_deriv_c_j = Eigen::MatrixX3d::Zero(num_collisions, 3);
            Eigen::MatrixX3d constraint_deriv_theta_j = Eigen::MatrixX3d::Zero(num_collisions, 3);
            Eigen::VectorXd constraint_deriv_t = Eigen::VectorXd::Zero(num_collisions);

            Eigen::VectorXd constraint = collision_constraint(vert_collisions);
            collision_constraint_deriv(trans_pos,
                                       trans_vel,
                                       angle,
                                       angle_vel,
                                       i,
                                       it->first,
                                       vert_collisions,
                                       constraint_deriv_t,
                                       constraint_deriv_theta_i,
                                       constraint_deriv_c_i,
                                       constraint_deriv_theta_j,
                                       constraint_deriv_c_j);

            // Check whether each collision violates the constraint
            // Add a penalty force to the rigid body for each collision that violates the constraint
            for (int k = 0; k < constraint.size(); k++)
            {
                uint body_j_id = vert_collisions[k].collidied_body;

                if (constraint(k) < 0 && constraint_deriv_t(k) < 0)
                {
                    F_trans.segment(3 * i, 3) += params_.penaltyStiffness * constraint(k) * constraint_deriv_c_i.row(k).transpose();
                    F_angle.segment(3 * i, 3) += params_.penaltyStiffness * constraint(k) * constraint_deriv_theta_i.row(k).transpose();

                    F_trans.segment(3 * body_j_id, 3) += params_.penaltyStiffness * constraint(k) * constraint_deriv_c_j.row(k).transpose();
                    F_angle.segment(3 * body_j_id, 3) += params_.penaltyStiffness * constraint(k) * constraint_deriv_theta_j.row(k).transpose();
                }
                else if (constraint(k) < 0 && constraint_deriv_t(k) >= 0)
                {
                    F_trans.segment(3 * i, 3) += params_.penaltyStiffness * params_.coefficientOfRestitution * constraint(k) * constraint_deriv_c_i.row(k).transpose();
                    F_angle.segment(3 * i, 3) += params_.penaltyStiffness * params_.coefficientOfRestitution * constraint(k) * constraint_deriv_theta_i.row(k).transpose();

                    F_trans.segment(3 * body_j_id, 3) += params_.penaltyStiffness * params_.coefficientOfRestitution * constraint(k) * constraint_deriv_c_j.row(k).transpose();
                    F_angle.segment(3 * body_j_id, 3) += params_.penaltyStiffness * params_.coefficientOfRestitution * constraint(k) * constraint_deriv_theta_j.row(k).transpose();
                }
            }
        }
    }
}