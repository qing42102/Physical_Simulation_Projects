#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

/*
    Add the force due to gravity to the force vector F
    Add the Hessian of the gravity force to the Hessian matrix H
    V = -\sum_i m_i * g * y_i
    F = [0, \sum_i m_i * g]
    H = [0, 0, 0, 0]
*/
void gravity_force(const Eigen::VectorXd &q,
                   Eigen::VectorXd &F,
                   Eigen::SparseMatrix<double> &H)
{

    for (uint i = 0; i < particles_.size(); i++)
    {
        if (!particles_[i].fixed)
        {
            F(2 * i + 1) += params_.gravityG * particles_[i].mass;
        }
    }
}

/*
    Add the force due to spring to the force vector F
    Add the Hessian of the spring force to the Hessian matrix H
    V = \sum_{ij} \frac{1}{2} k_{ij} (\Vert p_i - p_j \Vert - L_{ij})^2
    F = \sum_{ij} -k_{ij} (\Vert p_i - p_j \Vert - L_{ij}) \frac{p_i - p_j}{\Vert p_i - p_j \Vert}
    H = \sum_{ij} k_{ij} \left(\frac{(p_i - p_j) (p_i - p_j)^T}{\Vert p_i - p_j \Vert^2} +
        (\Vert p_i - p_j \Vert - L_{ij}) \left(\frac{\mathbf{I}}{\Vert p_i - p_j \Vert} - \frac{(p_i - p_j) (p_i - p_j)^T}{\Vert p_i - p_j \Vert^3} \right) \right)
*/
void spring_force(const Eigen::VectorXd &q,
                  Eigen::VectorXd &F,
                  Eigen::SparseMatrix<double> &H)
{
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(16 * particles_.size());

    for (uint i = 0; i < connectors_.size(); i++)
    {
        Spring *spring = dynamic_cast<Spring *>(connectors_[i]);

        double L = spring->restlen;
        double k = spring->stiffness / L;

        int p1_index = connectors_[i]->p1;
        int p2_index = connectors_[i]->p2;

        Eigen::Vector2d pos1 = q.segment(2 * p1_index, 2);
        Eigen::Vector2d pos2 = q.segment(2 * p2_index, 2);

        double dist = (pos1 - pos2).norm();

        Eigen::Vector2d pos_unit = (pos1 - pos2) / dist;

        // Local Hessian matrix
        Eigen::Matrix2d H_value = k * (pos_unit * pos_unit.transpose() + (dist - L) * (Eigen::MatrixXd::Identity(2, 2) / dist - pos_unit * pos_unit.transpose() / dist));

        // Local force vector
        Eigen::VectorXd F_value = k * (dist - L) * pos_unit;

        if (!particles_[p1_index].fixed)
        {
            // Corresponding index in the force vector
            // The force is negative because it is applied to the first particle
            F.segment(2 * p1_index, 2) += -F_value;

            // Corresponding index in the Hessian matrix
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index + j, 2 * p1_index + k, -H_value(j, k)));
                }
            }
        }
        if (!particles_[p2_index].fixed)
        {
            F.segment(2 * p2_index, 2) += F_value;

            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index + j, 2 * p2_index + k, -H_value(j, k)));
                }
            }
        }
        // Cross Hessian terms
        if (!particles_[p1_index].fixed && !particles_[p2_index].fixed)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index + j, 2 * p2_index + k, H_value(j, k)));
                    triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index + j, 2 * p1_index + k, H_value(j, k)));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> H_spring(2 * particles_.size(), 2 * particles_.size());
    H_spring.setFromTriplets(triplet_list.begin(), triplet_list.end());

    H += H_spring;
}

/*
    Add the force due to the floor
    F = 0 if y_i <= -0.5
    F = m_i * (-v_i^2 / (-0.5 - y_i)) if -0.5 < y_i <= -0.45
*/
void floor_force(const Eigen::VectorXd &q,
                 Eigen::VectorXd &F,
                 Eigen::SparseMatrix<double> &H)
{
    for (uint i = 0; i < particles_.size(); i++)
    {
        if (particles_[i].pos(1) <= -0.5 && !particles_[i].fixed)
        {
            F(2 * i + 1) = 0;
        }
        else if (particles_[i].pos(1) > -0.5 && particles_[i].pos(1) <= -0.45 && !particles_[i].fixed)
        {
            F(2 * i + 1) = 0;
            F(2 * i + 1) += particles_[i].mass * (pow(particles_[i].vel(1), 2) / (0.5 + particles_[i].pos(1)));
        }
    }
}

/*
    Add the force due to viscous damping to the force vector F
    Add the Hessian of the viscous damping force to the Hessian matrix H
    F = k_damp (q_2^{i} - q_2^{i-1} - q_1^{i} - q_1^{i-1}) / h
    H = k_damp / h * -I
*/
void viscous_damping(const Eigen::VectorXd &q,
                     const Eigen::VectorXd &qprev,
                     Eigen::VectorXd &F,
                     Eigen::SparseMatrix<double> &H)
{
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(4 * particles_.size());

    for (uint i = 0; i < connectors_.size(); i++)
    {
        int p1_index = connectors_[i]->p1;
        int p2_index = connectors_[i]->p2;

        Eigen::Vector2d pos1 = q.segment(2 * p1_index, 2);
        Eigen::Vector2d pos2 = q.segment(2 * p2_index, 2);

        Eigen::Vector2d pos1_prev = qprev.segment(2 * p1_index, 2);
        Eigen::Vector2d pos2_prev = qprev.segment(2 * p2_index, 2);

        if (!particles_[p1_index].fixed)
        {
            // Corresponding index in the force vector
            F.segment(2 * p1_index, 2) += params_.dampingStiffness * (pos2 - pos2_prev - pos1 + pos1_prev) / params_.timeStep;

            // Corresponding index in the Hessian matrix
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index, 2 * p1_index, -params_.dampingStiffness / params_.timeStep));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index + 1, 2 * p1_index + 1, -params_.dampingStiffness / params_.timeStep));
        }
        if (!particles_[p2_index].fixed)
        {
            F.segment(2 * p2_index, 2) += params_.dampingStiffness * (pos1 - pos1_prev - pos2 + pos2_prev) / params_.timeStep;

            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index, 2 * p2_index, -params_.dampingStiffness / params_.timeStep));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index + 1, 2 * p2_index + 1, -params_.dampingStiffness / params_.timeStep));
        }
        // Cross Hessian terms
        if (!particles_[p1_index].fixed && !particles_[p2_index].fixed)
        {
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index, 2 * p2_index, params_.dampingStiffness / params_.timeStep));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index, 2 * p1_index, params_.dampingStiffness / params_.timeStep));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p1_index + 1, 2 * p2_index + 1, params_.dampingStiffness / params_.timeStep));
            triplet_list.push_back(Eigen::Triplet<double>(2 * p2_index + 1, 2 * p1_index + 1, params_.dampingStiffness / params_.timeStep));
        }
    }

    Eigen::SparseMatrix<double> H_damping(2 * particles_.size(), 2 * particles_.size());
    H_damping.setFromTriplets(triplet_list.begin(), triplet_list.end());

    H += H_damping;
}

void computeForceAndHessian(const Eigen::VectorXd &q,
                            const Eigen::VectorXd &qprev,
                            const Eigen::VectorXd &qdot,
                            Eigen::VectorXd &F,
                            Eigen::SparseMatrix<double> &H)
{
    // Compute the total force and Hessian for all potentials in the system
    // This function should respect the booleans in params_ to allow the user
    // to toggle on and off individual force types.

    // Clear the force vector and Hessian matrix for the new step
    F.setZero();
    H.setZero();

    if (params_.gravityEnabled)
    {
        std::cout << "Computing gravitational force\n";
        gravity_force(q, F, H);
    }
    if (params_.springsEnabled)
    {
        std::cout << "Computing spring force\n";
        spring_force(q, F, H);
    }
    if (params_.dampingEnabled)
    {
        std::cout << "Computing viscous damping force\n";
        viscous_damping(q, qprev, F, H);
    }
    // Floor force is added last to ensure it overwrites the other forces
    if (params_.floorEnabled)
    {
        std::cout << "Computing floor force\n";
        floor_force(q, F, H);
    }
}