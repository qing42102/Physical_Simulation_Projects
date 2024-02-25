#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

double getTotalParticleMass(int idx)
{
    double mass = particles_[idx].mass;
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        if ((*it)->p1 == idx || (*it)->p2 == idx)
            mass += 0.5 * (*it)->mass;
    }
    return mass;
}

void processGravityForce(Eigen::VectorXd &F)
{
    int nparticles = (int)particles_.size();
    for (int i = 0; i < nparticles; i++)
    {
        if (!particles_[i].fixed)
        {
            F[2 * i + 1] += params_.gravityG * getTotalParticleMass(i);
        }
    }
}

void processSpringForce(const Eigen::VectorXd &q, Eigen::VectorXd &F, std::vector<Eigen::Triplet<double>> &H)
{
    int nsprings = (int)connectors_.size();

    for (int i = 0; i < nsprings; i++)
    {
        Spring &s = *(Spring *)connectors_[i];
        Eigen::Vector2d p1 = q.segment<2>(2 * s.p1);
        Eigen::Vector2d p2 = q.segment<2>(2 * s.p2);
        double dist = (p2 - p1).norm();
        Eigen::Vector2d localF = s.stiffness * (dist - s.restlen) / dist * (p2 - p1);
        F.segment<2>(2 * s.p1) += localF;
        F.segment<2>(2 * s.p2) -= localF;

        Eigen::Matrix2d I;
        I << 1, 0, 0, 1;
        Eigen::Matrix2d localH = s.stiffness * (1.0 - s.restlen / dist) * I;
        localH += s.stiffness * s.restlen * (p2 - p1) * (p2 - p1).transpose() / dist / dist / dist;

        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                H.push_back(Eigen::Triplet<double>(2 * s.p1 + j, 2 * s.p1 + k, localH.coeff(j, k)));
                H.push_back(Eigen::Triplet<double>(2 * s.p2 + j, 2 * s.p2 + k, localH.coeff(j, k)));
                H.push_back(Eigen::Triplet<double>(2 * s.p1 + j, 2 * s.p2 + k, -localH.coeff(j, k)));
                H.push_back(Eigen::Triplet<double>(2 * s.p2 + j, 2 * s.p1 + k, -localH.coeff(j, k)));
            }
    }
}

void processDampingForce(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, std::vector<Eigen::Triplet<double>> &H)
{
    int nsprings = (int)connectors_.size();

    for (int i = 0; i < nsprings; i++)
    {
        Spring &s = *(Spring *)connectors_[i];
        Eigen::Vector2d p1 = q.segment<2>(2 * s.p1);
        Eigen::Vector2d p2 = q.segment<2>(2 * s.p2);
        Eigen::Vector2d p1prev = qprev.segment<2>(2 * s.p1);
        Eigen::Vector2d p2prev = qprev.segment<2>(2 * s.p2);

        Eigen::Vector2d relvel = (p2 - p2prev) / params_.timeStep - (p1 - p1prev) / params_.timeStep;
        Eigen::Vector2d localF = params_.dampingStiffness * relvel;
        F.segment<2>(2 * s.p1) += localF;
        F.segment<2>(2 * s.p2) -= localF;

        Eigen::Matrix2d I;
        I << 1, 0, 0, 1;
        Eigen::Matrix2d localH = params_.dampingStiffness * I / params_.timeStep;

        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                H.push_back(Eigen::Triplet<double>(2 * s.p1 + j, 2 * s.p1 + k, localH.coeff(j, k)));
                H.push_back(Eigen::Triplet<double>(2 * s.p2 + j, 2 * s.p2 + k, localH.coeff(j, k)));
                H.push_back(Eigen::Triplet<double>(2 * s.p1 + j, 2 * s.p2 + k, -localH.coeff(j, k)));
                H.push_back(Eigen::Triplet<double>(2 * s.p2 + j, 2 * s.p1 + k, -localH.coeff(j, k)));
            }
    }
}

void processFloorForce(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, std::vector<Eigen::Triplet<double>> &H)
{
    int nparticles = particles_.size();

    double basestiffness = 10000;
    double basedrag = 1000.0;

    for (int i = 0; i < nparticles; i++)
    {
        if (q[2 * i + 1] < -0.5 && !particles_[i].fixed)
        {
            double vel = (q[2 * i + 1] - qprev[2 * i + 1]) / params_.timeStep;
            double dist = -0.5 - q[2 * i + 1];

            F[2 * i + 1] += basestiffness * dist - basedrag * dist * vel;

            H.push_back(Eigen::Triplet<double>(2 * i + 1, 2 * i + 1, basestiffness - 0.5 * basedrag / params_.timeStep + basedrag * qprev[2 * i + 1] / params_.timeStep - 2.0 * basedrag * q[2 * i + 1] / params_.timeStep));
        }
    }
}

void processBendingForce(const Eigen::VectorXd &q, Eigen::VectorXd &F)
{
    // TODO
    // Implement the bending energy and force for ropes
}

void computeForceAndHessian(const Eigen::VectorXd &q, const Eigen::VectorXd &qprev, Eigen::VectorXd &F, Eigen::SparseMatrix<double> &H)
{
    F.resize(q.size());
    F.setZero();
    H.resize(q.size(), q.size());
    H.setZero();

    std::vector<Eigen::Triplet<double>> Hcoeffs;
    if (params_.gravityEnabled)
        processGravityForce(F);
    if (params_.springsEnabled)
        processSpringForce(q, F, Hcoeffs);
    if (params_.dampingEnabled)
        processDampingForce(q, qprev, F, Hcoeffs);
    if (params_.floorEnabled)
        processFloorForce(q, qprev, F, Hcoeffs);
    if (params_.bendingEnabled)
        processBendingForce(q, F);

    H.setFromTriplets(Hcoeffs.begin(), Hcoeffs.end());
}