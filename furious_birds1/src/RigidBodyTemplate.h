#ifndef RIGIDBODYTEMPLATE_H
#define RIGIDBODYTEMPLATE_H

#include <string>
#include <Eigen/Core>
#include <set>
#include <vector>
#include "loadObjwithMaterial.h"
#include <glm/glm.hpp>

class SignedDistanceField;

class RigidBodyTemplate
{
public:
    RigidBodyTemplate(const std::string &meshFilename, double scale);
    ~RigidBodyTemplate();

    double getVolume() const { return volume_; }
    const Eigen::Matrix3d getInertiaTensor() const { return inertiaTensor_; }

    double getBoundingRadius() const { return radius_; }
    const Eigen::MatrixX3d &getVerts() const { return V; }
    const Eigen::MatrixX3i &getFaces() const { return F; }
    const Eigen::MatrixXd &getUVcoords() const {return uv_coords;}
    const Eigen::MatrixXi &getFTC() const {return FTC;}
    Material material;
    std::string folder_path;

private:
    RigidBodyTemplate(const RigidBodyTemplate &other) = delete;
    RigidBodyTemplate &operator=(const RigidBodyTemplate &other) = delete;

    void initialize();

    void computeVolume();
    Eigen::Vector3d computeCenterOfMass();
    void computeInertiaTensor();
    void populate_uv_coords();

    Eigen::MatrixX3d V;
    Eigen::MatrixX3i F;
    Eigen::MatrixXd TC, N;
    Eigen::MatrixXi FTC,FN;
    Eigen::MatrixXd uv_coords;

    double volume_;
    double radius_;
    Eigen::Matrix3d inertiaTensor_;
};

extern std::vector<RigidBodyTemplate *> templates_;

#endif // RIGIDBODYTEMPLATE_H
