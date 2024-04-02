#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include "loadObjwithMaterial.h"
#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include <stb_image.h>
#include <filesystem>
namespace fs = std::__fs::filesystem;
using namespace std;
// Assuming you have a structure to hold material properties
std::string getPathFolderName(const std::string& filePath) {
    fs::path p(filePath);
    // Get the parent path and then its filename component, which is the directory name
    return p.parent_path().filename().string();
}

void loadTextureImage(const std::string& filename, std::vector<glm::vec3>& colorsTex, int& width, int& height) {
    int channels;
    unsigned char* imgData = stbi_load(filename.c_str(), &width, &height, &channels, 3); // Force RGB
    if (imgData) {
        colorsTex.resize(width * height);
        for (int i = 0; i < width * height; ++i) {
            colorsTex[i] = glm::vec3(imgData[i * 3] / 255.0f, imgData[i * 3 + 1] / 255.0f, imgData[i * 3 + 2] / 255.0f);
        }
        stbi_image_free(imgData);
    } else {
        // Handle error
        std::cout << "could not load texture map" << std::endl;
    }
}

// std::map<std::string, Material> parseMTL(const std::string& filePath) {
//     std::map<std::string, Material> materials;
//     std::ifstream file(filePath);
//     std::string line;
//     Material currentMaterial;
//     while (std::getline(file, line)) {
//         std::istringstream iss(line);
//         std::string identifier;
//         iss >> identifier;
//         if (identifier == "newmtl") {
//             if (!currentMaterial.name.empty()) {
//                 materials[currentMaterial.name] = currentMaterial;
//                 currentMaterial = Material(); // Reset for the next material
//             }
//             iss >> currentMaterial.name;
//         } else if (identifier == "map_Kd") {
//             iss >> currentMaterial.map_Kd;
//             std::cout << "mtl path: " << currentMaterial.map_Kd << std:: endl;
//         }
//         // Handle other properties if you want
//     }
//     if (!currentMaterial.name.empty()) {
//         materials[currentMaterial.name] = currentMaterial; // Add the last material
//     }
//     return materials;
// }

Material parseMTL(const std::string& filePath) {
    std::ifstream file(filePath);
    std::string line;
    Material currentMaterial;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string identifier;
        iss >> identifier;
        if (identifier == "newmtl") {
            iss >> currentMaterial.name;
        } else if (identifier == "map_Kd") {
            iss >> currentMaterial.map_Kd;
            std::cout << "mtl path: " << currentMaterial.map_Kd << std:: endl;
            break;
        }
        // Handle other properties if you want
    }
    return currentMaterial;
    
}