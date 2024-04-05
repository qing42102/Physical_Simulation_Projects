#include <vector>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/signed_distance.h>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"

Eigen::MatrixXd bounding_box(RigidBodyInstance *body)
{
    double radius = body->getTemplate().getBoundingRadius();
    Eigen::Vector3d center = body->c;
    Eigen::Matrix3d rotation = VectorMath::rotationMatrix(body->theta);

    // Construct a cubic box around the body with side length 2 * radius
    // Box is stored as a 3x8 matrix where each column is a vertex of the box
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

bool bounding_box_overlap(Eigen::MatrixXd box1, Eigen::MatrixXd box2)
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

void detect_collisions()
{
    std::vector<Eigen::MatrixXd> boxes(bodies_.size());
    for (int i = 0; i < bodies_.size(); i++)
    {
        Eigen::MatrixXd box = bounding_box(bodies_[i]);
        boxes[i] = box;
    }

    for (int i = 0; i < bodies_.size(); i++)
    {
        for (int j = i + 1; j < bodies_.size(); j++)
        {
            // Broad-phase collision detection using bounding boxes
            if (bounding_box_overlap(boxes[i], boxes[j]))
            {
                // Perform narrow-phase collision detection
                Eigen::Matrix3d vertices = bodies_[i]->getTemplate().getVerts();
            }
        }
    }
}