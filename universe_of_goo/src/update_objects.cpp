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
    Delete springs that are connected to a particle that is about to be deleted
    @param particle_index Index of the particle to be deleted
 */
void delete_unconnected_springs(int particle_index)
{
    std::vector<Connector *> new_connectors_;
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        Spring *spring = dynamic_cast<Spring *>(*it);
        // Delete springs connected to this particle
        if (spring->p1 == particle_index || spring->p2 == particle_index)
        {
            delete spring;
        }
        // Keep the spring in the new list if it's not connected to the deleted particle
        else
        {
            // Update the indices of the springs that are connected to particles with higher indices
            if (spring->p1 > particle_index)
                spring->p1--;
            if (spring->p2 > particle_index)
                spring->p2--;
            new_connectors_.push_back(spring);
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
    Delete springs that are too close to a saw
*/
void delete_sawed_springs()
{
    std::vector<Connector *> new_connectors_;
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        Spring *spring = dynamic_cast<Spring *>(*it);
        Eigen::Vector2d pos1 = particles_[spring->p1].pos;
        Eigen::Vector2d pos2 = particles_[spring->p2].pos;

        bool delete_spring = false;
        for (std::vector<Saw>::iterator saw = saws_.begin(); saw != saws_.end(); ++saw)
        {
            // Delete springs that are too close to a saw
            // The distance between the saw's center and the line formed by the spring's endpoints is less than the saw's radius
            double dist = point_to_finite_line_dist(saw->pos, pos1, pos2);
            if (dist < saw->radius)
            {
                delete_spring = true;
                break;
            }
        }

        if (delete_spring)
        {
            delete spring;
        }
        // Keep the spring in the new list if it doesn't touch a saw
        else
        {
            new_connectors_.push_back(spring);
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
            delete_unconnected_springs(i);
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
        // Delete particles that are offscreen
        if (particles_[i].pos[1] < -1.0 || particles_[i].pos[1] > 1 || particles_[i].pos[0] < -2.0 || particles_[i].pos[0] > 2.0)
        {
            delete_unconnected_springs(i);
        }
        // Keep the particle in a new list if it's not offscreen
        else
        {
            new_particles_.push_back(particles_[i]);
        }
    }

    particles_ = new_particles_;
}