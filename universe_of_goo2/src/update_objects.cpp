#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

void add_connectors(int newid, Eigen::Vector2d newpos)
{
    int numparticles = particles_.size() - 1;

    for (int i = 0; i < numparticles; i++)
    {
        if (particles_[i].inert)
            continue;
        Eigen::Vector2d pos = particles_[i].pos;
        double dist = (pos - newpos).norm();
        if (dist <= params_.maxSpringDist)
        {
            // Also support rigid and flexible rods
            if (params_.connectorType == SimParameters::CT_SPRING)
            {
                connectors_.push_back(new Spring(newid, i, 0, params_.springStiffness / dist, dist, true));
            }
            else if (params_.connectorType == SimParameters::CT_RIGIDROD)
            {
                connectors_.push_back(new RigidRod(newid, i, 0, dist));
            }
            else if (params_.connectorType == SimParameters::CT_FLEXROD)
            {
                int num_segments = std::max(2, params_.rodSegments);
                Eigen::Vector2d segment = (pos - newpos) / num_segments;
                double segment_dist = segment.norm();

                double flex_rod_mass = params_.rodDensity * segment_dist;
                double flex_rod_stiffness = params_.rodStretchingStiffness / segment_dist;

                for (int i = 1; i < num_segments; i++)
                {
                    particles_.push_back(Particle(newpos + i * segment, 0, false, true));
                    connectors_.push_back(new Spring(newid + i - 1, newid + i, flex_rod_mass, flex_rod_stiffness, segment_dist, false));
                }
                connectors_.push_back(new Spring(newid + num_segments - 1, i, flex_rod_mass, flex_rod_stiffness, segment_dist, false));
            }
        }
    }
}

void addParticle(double x, double y)
{
    Eigen::Vector2d newpos(x, y);
    double mass = params_.particleMass;
    if (params_.particleFixed)
        mass = std::numeric_limits<double>::infinity();

    int newid = particles_.size();
    particles_.push_back(Particle(newpos, mass, params_.particleFixed, false));

    add_connectors(newid, newpos);
}

void addSaw(double x, double y)
{
    saws_.push_back(Saw(Eigen::Vector2d(x, y), params_.sawRadius));
}

void buildConfiguration(Eigen::VectorXd &q, Eigen::VectorXd &lambda, Eigen::VectorXd &qdot)
{
    int ndofs = 2 * particles_.size();
    q.resize(ndofs);
    qdot.resize(ndofs);

    for (int i = 0; i < (int)particles_.size(); i++)
    {
        q.segment<2>(2 * i) = particles_[i].pos;
        qdot.segment<2>(2 * i) = particles_[i].vel;
    }

    // TODO
    //  Pack the Lagrange multiplier degrees of freedom into the global configuration vector lambda
}

void unbuildConfiguration(const Eigen::VectorXd &q, const Eigen::VectorXd &lambda, const Eigen::VectorXd &qdot)
{
    int ndofs = q.size();
    assert(ndofs == int(2 * particles_.size()));

    for (int i = 0; i < ndofs / 2; i++)
    {
        particles_[i].pos = q.segment<2>(2 * i);
        particles_[i].vel = qdot.segment<2>(2 * i);
    }

    // TODO
    // Unpack the configurational Lagrange multipliers back into the RigidRods
}

/*
    Delete springs that are connected to a particle that is about to be deleted
    @param particle_index Index of the particle to be deleted
 */
void delete_unconnected_connectors(int particle_index)
{
    std::vector<Connector *> new_connectors_;
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        // Delete springs connected to this particle
        if ((*it)->p1 == particle_index || (*it)->p2 == particle_index)
        {
            delete *it;
        }
        // Keep the connector in the new list if it's not connected to the deleted particle
        else
        {
            // Update the indices of the springs that are connected to particles with higher indices
            if ((*it)->p1 > particle_index)
                (*it)->p1--;
            if ((*it)->p2 > particle_index)
                (*it)->p2--;
            new_connectors_.push_back(*it);
        }
    }
    connectors_ = new_connectors_;
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
    Delete springs or rigid rods that are too close to a saw
*/
void delete_sawed_connectors()
{
    std::vector<Connector *> new_connectors_;
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        Eigen::Vector2d pos1 = particles_[(*it)->p1].pos;
        Eigen::Vector2d pos2 = particles_[(*it)->p2].pos;

        bool delete_connector = false;
        for (std::vector<Saw>::iterator saw = saws_.begin(); saw != saws_.end(); ++saw)
        {
            // Delete connectors that are too close to a saw
            // The distance between the saw's center and the line formed by the connector's endpoints is less than the saw's radius
            double dist = point_to_finite_line_dist(saw->pos, pos1, pos2);
            if (dist < saw->radius)
            {
                delete_connector = true;
                break;
            }
        }

        if (delete_connector)
        {
            delete *it;
        }
        // Keep the connector in the new list if it doesn't touch a saw
        else
        {
            new_connectors_.push_back(*it);
        }
    }
    connectors_ = new_connectors_;
}

/*
    Delete particles that are too close to a saw
*/
void delete_sawed_particles()
{
    std::vector<Particle, Eigen::aligned_allocator<Particle>> new_particles_;

    for (uint i = 0; i < particles_.size(); i++)
    {
        bool delete_particle = false;
        for (std::vector<Saw>::iterator it = saws_.begin(); it != saws_.end(); ++it)
        {
            // Delete this particle that is too close to a saw
            // The distance between the particle and the saw is less than the saw's radius
            if ((particles_[i].pos - it->pos).norm() < it->radius)
            {
                delete_particle = true;
                break;
            }
        }

        if (delete_particle)
        {
            delete_unconnected_connectors(i);
        }
        // Keep the particle in the new list if it doesn't touch a saw
        else
        {
            new_particles_.push_back(particles_[i]);
        }
    }

    particles_ = new_particles_;
}

void pruneOverstrainedSprings()
{
    std::vector<Connector *> new_connectors_;

    // Delete springs that have too high strain
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        if ((*it)->getType() == SimParameters::CT_SPRING)
        {
            Spring *spring = dynamic_cast<Spring *>(*it);
            Eigen::Vector2d pos1 = particles_[spring->p1].pos;
            Eigen::Vector2d pos2 = particles_[spring->p2].pos;

            double dist = (pos1 - pos2).norm();
            double strain = (dist - spring->restlen) / spring->restlen;

            if (strain > params_.maxSpringStrain)
            {
                delete spring;
            }
            // Keep the spring in the new list if it's not overstrained
            else
            {
                new_connectors_.push_back(spring);
            }
        }
        else
        {
            new_connectors_.push_back(*it);
        }
    }

    connectors_ = new_connectors_;
}

/*
    Delete particles that are offscreen
*/
void delete_offscreen_particles()
{
    std::vector<Particle, Eigen::aligned_allocator<Particle>> new_particles_;

    for (uint i = 0; i < particles_.size(); i++)
    {
        if (particles_[i].pos[1] < -1.0 || particles_[i].pos[1] > 1 || particles_[i].pos[0] < -2.0 || particles_[i].pos[0] > 2.0)
        {
            delete_unconnected_connectors(i);
        }
        // Keep the particle in a new list if it's not offscreen
        else
        {
            new_particles_.push_back(particles_[i]);
        }
    }

    particles_ = new_particles_;
}