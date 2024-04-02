#include "polyscope/polyscope.h"

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"


#include <iostream>
#include <unordered_set>
#include <utility>
#include <deque>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "loadObjwithMaterial.h"
#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"
#include <fstream>
#include <glm/glm.hpp>
#include "misc/cpp/imgui_stdlib.h"

#include "numerical_integration.cpp"
#include "update_objects.cpp"

bool running_;
double time_;
SimParameters params_;
std::string sceneFile_;

std::vector<RigidBodyTemplate *> templates_;
std::vector<RigidBodyInstance *> bodies_;

Eigen::MatrixXd renderQ;
Eigen::MatrixXi renderF;

void updateRenderGeometry()
{
    int totverts = 0;
    int totfaces = 0;
    for (RigidBodyInstance *rbi : bodies_)
    {
        totverts += rbi->getTemplate().getVerts().rows();
        totfaces += rbi->getTemplate().getFaces().rows();
    }
    renderQ.resize(totverts, 3);
    renderF.resize(totfaces, 3);
    int voffset = 0;
    int foffset = 0;
    for (RigidBodyInstance *rbi : bodies_)
    {
        int nverts = rbi->getTemplate().getVerts().rows();
        for (int i = 0; i < nverts; i++)
            renderQ.row(voffset + i) = (rbi->c + VectorMath::rotationMatrix(rbi->theta) * rbi->getTemplate().getVerts().row(i).transpose()).transpose();
        int nfaces = rbi->getTemplate().getFaces().rows();
        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                renderF(foffset + i, j) = rbi->getTemplate().getFaces()(i, j) + voffset;
            }
        }
        voffset += nverts;
        foffset += nfaces;
    }
}

void loadScene()
{
    for (RigidBodyInstance *rbi : bodies_)
        delete rbi;
    for (RigidBodyTemplate *rbt : templates_)
        delete rbt;
    bodies_.clear();
    templates_.clear();

    std::string prefix;
    std::string scenefname = std::string("scenes/") + sceneFile_;
    std::ifstream ifs(scenefname);
    if (!ifs)
    {
        // run from the build directory?
        prefix = "../";
        std::string fullname = prefix + scenefname;
        ifs.open(fullname);
        if (!ifs)
        {
            prefix = "../../";
            fullname = prefix + scenefname;
            ifs.open(fullname);
            if (!ifs)
                return;
        }
    }

    int nbodies;
    ifs >> nbodies;
    for (int body = 0; body < nbodies; body++)
    {
        std::string meshname, folder_path;
        ifs >> meshname;
        meshname = prefix + std::string("meshes/") + meshname;
        folder_path = prefix + std::string("meshes/") + getPathFolderName(meshname);
        double scale;
        ifs >> scale;
        RigidBodyTemplate *rbt = new RigidBodyTemplate(meshname, scale);
        rbt->folder_path = folder_path;
        double rho;
        ifs >> rho;
        Eigen::Vector3d c, theta, cvel, w;
        for (int i = 0; i < 3; i++)
            ifs >> c[i];
        for (int i = 0; i < 3; i++)
            ifs >> theta[i];
        for (int i = 0; i < 3; i++)
            ifs >> cvel[i];
        for (int i = 0; i < 3; i++)
            ifs >> w[i];
        RigidBodyInstance *rbi = new RigidBodyInstance(*rbt, c, theta, cvel, w, rho);
        templates_.push_back(rbt);
        bodies_.push_back(rbi);
    }
}

void initSimulation()
{
    time_ = 0;
    loadScene();
    updateRenderGeometry();
}

void simulateOneStep()
{
    system("clear");

    // Create configurational vectors
    Eigen::VectorXd trans_pos, trans_vel, angle, angle_vel;
    buildConfiguration(trans_pos, trans_vel, angle, angle_vel);
    // Use them for one step of time integration
    numericalIntegration(trans_pos, trans_vel, angle, angle_vel);
    // Unpack the DOFs back into the particles for rendering
    unbuildConfiguration(trans_pos, trans_vel, angle, angle_vel);

    // Time advances
    time_ += params_.timeStep;
    std::cout << "Time: " << time_ << "\n";
}

void callback()
{
    ImGui::SetNextWindowSize(ImVec2(500., 0.));
    ImGui::Begin("UI", nullptr);

    if (ImGui::Button("Recenter Camera", ImVec2(-1, 0)))
    {
        polyscope::view::resetCameraToHomeView();
    }

    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Run/Pause Sim", ImVec2(-1, 0)))
        {
            running_ = !running_;
        }
        if (ImGui::Button("Reset Sim", ImVec2(-1, 0)))
        {
            running_ = false;
            initSimulation();
        }
    }
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputText("Filename", &sceneFile_);
        if (ImGui::Button("Load Scene", ImVec2(-1, 0)))
        {
            loadScene();
            initSimulation();
        }
    }
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Timestep", &params_.timeStep);
        ImGui::InputDouble("Newton Tolerance", &params_.NewtonTolerance);
        ImGui::InputInt("Newton Max Iters", &params_.NewtonMaxIters);
    }
    if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Gravity Enabled", &params_.gravityEnabled);
        ImGui::InputDouble("Gravity G", &params_.gravityG);
    }

    ImGui::End();
}

int main(int argc, char **argv)
{
    polyscope::view::setWindowSize(1600, 800);
    polyscope::options::buildGui = false;
    polyscope::options::openImGuiWindowForUserCallback = false;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    polyscope::options::autocenterStructures = false;
    polyscope::options::autoscaleStructures = false;
    polyscope::options::maxFPS = -1;

    sceneFile_ = "cb1.scn";

    initSimulation();

    polyscope::init();

    polyscope::state::userCallback = callback;

    while (!polyscope::render::engine->windowRequestsClose())
    {
        if (running_)
            simulateOneStep();
        updateRenderGeometry();
        auto *surf = polyscope::registerSurfaceMesh("Bodies", renderQ, renderF);
        surf->setTransparency(0.9);
        for (RigidBodyTemplate* rbt : templates_){
            std::vector<glm::vec3> colorsTex;
            if(!rbt->material.map_Kd.empty()){
                int width, height;
                std::string material_abs_path = rbt->folder_path+ "/" + rbt->material.map_Kd;
                loadTextureImage(material_abs_path, colorsTex, width, height);
                // Add UV coordinates as a parameterization quantity if not done
                auto qParam = surf->addParameterizationQuantity("UV_0", rbt->getUVcoords());
                // Add the texture image as a color quantity using the loaded image data
                auto* texture = surf->addTextureColorQuantity("Texture_0", *qParam,
                width, height, colorsTex, polyscope::ImageOrigin::UpperLeft);
                texture->setEnabled(true);
            }

        }

        polyscope::frameTick();
    }

    return 0;
}
