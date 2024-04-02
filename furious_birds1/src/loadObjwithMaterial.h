#ifndef MATERIALINSTANCE_H
#define MATERIALINSTANCE_H

#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <glm/glm.hpp>


using namespace std;
using namespace Eigen;
// Assuming you have a structure to hold material properties
struct Material {
    float Ns; // Specular exponent
    Eigen::Vector3f Ka; // Ambient color
    Eigen::Vector3f Kd; // Diffuse color
    Eigen::Vector3f Ks; // Specular color
    float Ni; // Optical density
    float d; // Dissolve (transparency)
    int illum; // Illumination model
    std::string map_Kd; // Diffuse texture map
    std::string map_Bump; // Bump map
    std::string map_Ks; // Specular color texture map
    std::string name;

    // Constructor, getters, setters, etc.
    Material() : Ns(0), Ni(0), d(0), illum(0) {
        Ka = Eigen::Vector3f::Zero();
        Kd = Eigen::Vector3f::Zero();
        Ks = Eigen::Vector3f::Zero();
    }
};

Material parseMTL(const std::string& filePath);
void loadTextureImage(const std::string& filename, 
                        std::vector<glm::vec3>& colorsTex, 
                        int& width, int& height);
std::string getPathFolderName(const std::string& filePath);
#endif