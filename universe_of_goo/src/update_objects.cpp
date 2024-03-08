#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

void addParticle(double x, double y)
{
    Eigen::Vector2d newpos(x, y);
    double mass = params_.particleMass;
    if (params_.particleFixed)
        mass = std::numeric_limits<double>::infinity();

    int newid = particles_.size();
    particles_.push_back(Particle(newpos, mass, params_.particleFixed, false));

    // Connect particles to nearby ones with springs
    for (int i = 0; i < newid; i++)
    {
        double dist = (particles_[i].pos - newpos).norm();
        if (dist <= params_.maxSpringDist)
        {
            connectors_.push_back(new Spring(i, newid, 0, params_.springStiffness, dist, true));
        }
    }
}

void addSaw(double x, double y)
{
    saws_.push_back(Saw(Eigen::Vector2d(x, y), params_.sawRadius));
}

void buildConfiguration(Eigen::VectorXd &q, Eigen::VectorXd &qprev, Eigen::VectorXd &qdot)
{
    q.resize(2 * particles_.size());
    qprev.resize(2 * particles_.size());
    qdot.resize(2 * particles_.size());

    //  Pack the degrees of freedom and DOF velocities into global configuration vectors
    for (uint i = 0; i < particles_.size(); i++)
    {
        q.segment<2>(2 * i) = particles_[i].pos;
        qprev.segment<2>(2 * i) = particles_[i].prevpos;
        qdot.segment<2>(2 * i) = particles_[i].vel;
    }
}

void unbuildConfiguration(const Eigen::VectorXd &q, const Eigen::VectorXd &qdot)
{
    // Unpack the configurational position vectors back into the particles_ for rendering
    for (uint i = 0; i < particles_.size(); i++)
    {
        particles_[i].prevpos = particles_[i].pos;
        particles_[i].pos = q.segment<2>(2 * i);
        particles_[i].vel = qdot.segment<2>(2 * i);
    }
}

/*
    Calculate the distance between a point and an infinite line
    @param p The point
    @param q1 A point on the line
    @param q2 Another point on the line
    @return The distance between the point and the line

*/
double point_to_infinite_line_dist(const Eigen::Vector2d &p, const Eigen::Vector2d &q1, const Eigen::Vector2d &q2)
{
    Eigen::Vector2d line_vec = (q2 - q1) / (q2 - q1).norm();
    Eigen::Vector2d v = p - q1;
    double t = v.dot(line_vec);
    Eigen::Vector2d c = q1 + t * line_vec;

    return (p - c).norm();
}

/*
    Calculate the distance between a point and a finite line
    @param p The point
    @param q1 One end of the line
    @param q2 The other end of the line
    @return The distance between the point and the line
*/
double point_to_finite_line_dist(const Eigen::Vector2d &p, const Eigen::Vector2d &q1, const Eigen::Vector2d &q2)
{
    Eigen::Vector2d line_vec = (q2 - q1) / (q2 - q1).norm();
    Eigen::Vector2d v = p - q1;
    double t = std::max(0.0, std::min(1.0, v.dot(line_vec)));
    Eigen::Vector2d c = q1 + t * line_vec;

    return (p - c).norm();
}

/*
    Delete particles that are offscreen or too close to a saw
    @return A list of booleans indicating which particles are about to be deleted
*/
std::vector<bool> detect_particles_to_delete()
{
    std::vector<bool> delete_particle(particles_.size(), false);
    for (uint i = 0; i < particles_.size(); i++)
    {
        for (std::vector<Saw>::iterator it = saws_.begin(); it != saws_.end(); ++it)
        {
            // Delete this particle that is too close to a saw
            // The distance between the particle and the saw is less than the saw's radius
            if ((particles_[i].pos - it->pos).norm() < it->radius)
            {
                delete_particle[i] = true;
                break;
            }
        }

        if (delete_particle[i])
            continue;

        // Delete particles that are offscreen
        if (particles_[i].pos[1] < -1.0 ||
            particles_[i].pos[1] > 1 ||
            particles_[i].pos[0] < -2.0 ||
            particles_[i].pos[0] > 2.0)
        {
            delete_particle[i] = true;
        }
    }

    return delete_particle;
}

/*
    Delete connectors that are too close to a saw or connected to a particle that is about to be deleted
    @param delete_particles A list of booleans indicating which particles are about to be deleted
    @return A list of booleans indicating which connectors are about to be deleted
*/
std::vector<bool> detect_connectors_to_delete(const std::vector<bool> &delete_particles)
{
    std::vector<bool> delete_connector(connectors_.size(), false);
    for (uint i = 0; i < connectors_.size(); i++)
    {
        Eigen::Vector2d pos1 = particles_[connectors_[i]->p1].pos;
        Eigen::Vector2d pos2 = particles_[connectors_[i]->p2].pos;
        double maxx = std::max(pos1[0], pos2[0]);
        double minx = std::min(pos1[0], pos2[0]);
        double maxy = std::max(pos1[1], pos2[1]);
        double miny = std::min(pos1[1], pos2[1]);

        for (std::vector<Saw>::iterator saw = saws_.begin(); saw != saws_.end(); ++saw)
        {
            // Need to account for the saws that are horizontally or vertically aligned with the connector but not actually close to it
            // The point_to_finite_line_dist function would return a small distance because it's measuring the perpendicular distance to the line
            if (saw->pos[0] - saw->radius <= maxx &&
                saw->pos[0] + saw->radius >= minx &&
                saw->pos[1] - saw->radius <= maxy &&
                saw->pos[1] + saw->radius >= miny)
            {
                // Delete connectors that are too close to a saw
                // The distance between the saw's center and the line formed by the connector's endpoints is less than the saw's radius
                double dist = point_to_finite_line_dist(saw->pos, pos1, pos2);
                if (dist < saw->radius)
                {
                    delete_connector[i] = true;
                    break;
                }
            }
        }

        if (delete_connector[i])
            continue;

        // Delete connectors that are connected to a particle that is about to be deleted
        if (delete_particles[connectors_[i]->p1] || delete_particles[connectors_[i]->p2])
        {
            delete_connector[i] = true;
        }
        else
        {
            // Update the indices of the particles that are not deleted to the range [0, particles_.size() - 1]
            // The indices of the particles that are not deleted are reduced by the number of particles that are deleted before them
            int p1_deletion = std::count(delete_particles.begin(), delete_particles.begin() + connectors_[i]->p1, true);
            int p2_deletion = std::count(delete_particles.begin(), delete_particles.begin() + connectors_[i]->p2, true);

            connectors_[i]->p1 -= p1_deletion;
            connectors_[i]->p2 -= p2_deletion;
        }
    }

    return delete_connector;
}

void delete_objects()
{
    std::vector<bool> delete_particles = detect_particles_to_delete();
    std::vector<bool> delete_connectors = detect_connectors_to_delete(delete_particles);

    // Keep the particle in a new list if it's not deleted
    std::vector<Particle, Eigen::aligned_allocator<Particle>> new_particles_;
    for (uint i = 0; i < particles_.size(); i++)
    {
        if (!delete_particles[i])
        {
            new_particles_.push_back(particles_[i]);
        }
    }
    particles_ = new_particles_;

    // Keep the connector in a new list if it's not deleted
    std::vector<Connector *> new_connectors_;
    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (!delete_connectors[i])
        {
            new_connectors_.push_back(connectors_[i]);
        }
        else
        {
            delete connectors_[i];
        }
    }
    connectors_ = new_connectors_;

    std::cout << "Delete sawed objects\n";
}

void pruneOverstrainedSprings()
{
    std::vector<Connector *> new_connectors_;

    // Delete springs that have too high strain
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        Spring *spring = dynamic_cast<Spring *>(*it);
        Eigen::Vector2d pos1 = particles_[spring->p1].pos;
        Eigen::Vector2d pos2 = particles_[spring->p2].pos;

        double dist = (pos1 - pos2).norm();
        double strain = (dist - spring->restlen) / spring->restlen;

        if (spring->canSnap && strain > params_.maxSpringStrain)
        {
            delete spring;
        }
        // Keep the spring in the new list if it's not overstrained
        else
        {
            new_connectors_.push_back(spring);
        }
    }

    connectors_ = new_connectors_;
}
