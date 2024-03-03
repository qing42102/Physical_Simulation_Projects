#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "SceneObjects.h"

void add_hinges(const int &num_segments, const double &segment_len)
{
    // The indices of the springs added by flexible rods
    for (uint i = connectors_.size() - num_segments; i < connectors_.size() - 1; i++)
    {
        Spring *spring1 = dynamic_cast<Spring *>(connectors_[i]);
        Spring *spring2 = dynamic_cast<Spring *>(connectors_[i + 1]);

        double hinge_stiffness = 2 * params_.rodBendingStiffness / pow(segment_len, 2);
        // Add a bending hinge between each triplet of particles
        bendingStencils_.push_back(BendingStencil(spring1->p1, spring1->p2, spring2->p2, hinge_stiffness));

        spring1->associatedBendingStencils.insert(bendingStencils_.size() - 1);
        spring2->associatedBendingStencils.insert(bendingStencils_.size() - 1);
    }
}

void add_connectors(const int &newid, const Eigen::Vector2d &newpos)
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
                double segment_len = segment.norm();

                double flex_rod_mass = params_.rodDensity * segment_len;
                double flex_rod_stiffness = params_.rodStretchingStiffness / segment_len;

                // Mimic a flexible rod with a chain of particles connected by springs
                for (int j = 1; j < num_segments; j++)
                {
                    // The particles are equally spaced along the rod starting from the new particle
                    particles_.push_back(Particle(newpos + j * segment, 0, false, true));

                    if (j == 1)
                    {
                        // Connect the first particle of the flexible rod to the new particle
                        connectors_.push_back(new Spring(newid, particles_.size() - 1, flex_rod_mass, flex_rod_stiffness, segment_len, false));
                    }
                    else
                    {
                        // Connect the previous particle of the flexible rod to the current one
                        connectors_.push_back(new Spring(particles_.size() - 2, particles_.size() - 1, flex_rod_mass, flex_rod_stiffness, segment_len, false));
                    }
                }
                // The last particlee of the flexible rod is connected to the original particle
                connectors_.push_back(new Spring(particles_.size() - 1, i, flex_rod_mass, flex_rod_stiffness, segment_len, false));

                add_hinges(num_segments, segment_len);
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

        if (!delete_particle[i])
        {
            // Delete particles that are offscreen
            if (particles_[i].pos[1] < -1.0 || particles_[i].pos[1] > 1 || particles_[i].pos[0] < -2.0 || particles_[i].pos[0] > 2.0)
            {
                delete_particle[i] = true;
            }
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

        for (std::vector<Saw>::iterator saw = saws_.begin(); saw != saws_.end(); ++saw)
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

        if (!delete_connector[i])
        {
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
    }

    return delete_connector;
}

/*
    Delete bending stencils that are associated with connectors that are about to be deleted
    @param delete_spring A list of booleans indicating which connectors are about to be deleted
    @return A list of booleans indicating which bending stencils are about to be deleted
*/
std::vector<bool> detect_bending_to_delete(const std::vector<bool> &delete_spring)
{
    std::vector<bool> delete_bending(bendingStencils_.size(), false);
    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (delete_spring[i])
        {
            std::set<int> spring_bending_stencils = connectors_[i]->associatedBendingStencils;
            for (std::set<int>::iterator it = spring_bending_stencils.begin(); it != spring_bending_stencils.end(); ++it)
            {
                int index = *it;
                delete_bending[index] = true;
            }
        }
    }

    for (uint i = 0; i < connectors_.size(); i++)
    {
        if (!delete_spring[i])
        {
            std::set<int> spring_bending_stencils = connectors_[i]->associatedBendingStencils;
            std::set<int> new_spring_bending_stencils;
            for (std::set<int>::iterator it = spring_bending_stencils.begin(); it != spring_bending_stencils.end(); ++it)
            {
                // Update the indices of the bending stencils that are not deleted to the range [0, bendingStencils_.size() - 1]
                // The indices of the bending stencils that are not deleted are reduced by the number of bending stencils that are deleted before them
                int index = *it;
                if (!delete_bending[index])
                {
                    int bending_deletion = std::count(delete_bending.begin(), delete_bending.begin() + index, true);
                    new_spring_bending_stencils.insert(index - bending_deletion);
                }
            }

            connectors_[i]->associatedBendingStencils = new_spring_bending_stencils;
        }
    }

    return delete_bending;
}

void delete_objects()
{
    std::vector<bool> delete_particles = detect_particles_to_delete();
    std::vector<bool> delete_connectors = detect_connectors_to_delete(delete_particles);
    std::vector<bool> delete_bending = detect_bending_to_delete(delete_connectors);

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
    }
    connectors_ = new_connectors_;

    // Delete the bending stencils that are associated with the deleted connectors
    std::vector<BendingStencil> new_bending_stencils_;
    for (uint i = 0; i < bendingStencils_.size(); i++)
    {
        if (!delete_bending[i])
        {
            new_bending_stencils_.push_back(bendingStencils_[i]);
        }
    }
    bendingStencils_ = new_bending_stencils_;
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
        else
        {
            new_connectors_.push_back(*it);
        }
    }

    connectors_ = new_connectors_;
}
