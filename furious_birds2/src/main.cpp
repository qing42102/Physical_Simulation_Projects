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

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"
#include <fstream>
#include "misc/cpp/imgui_stdlib.h"
#include <igl/signed_distance.h>

bool running_;
double time_;
SimParameters params_;
std::string sceneFile_;
bool launch_;

std::vector<RigidBodyTemplate*> templates_;
std::vector<RigidBodyInstance*> bodies_;
RigidBodyTemplate* birdTemplate_;

Eigen::MatrixXd renderQ;
Eigen::MatrixXi renderF;
Eigen::MatrixXd groundV;
Eigen::MatrixXi groundF;

void updateRenderGeometry()
{            
    int totverts = 0;
    int totfaces = 0;
    for (RigidBodyInstance* rbi : bodies_)
    {
        totverts += rbi->getTemplate().getVerts().rows();
        totfaces += rbi->getTemplate().getFaces().rows();
    }
    renderQ.resize(totverts, 3);
    renderF.resize(totfaces, 3);
    int voffset = 0;
    int foffset = 0;
    for (RigidBodyInstance* rbi : bodies_)
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

    double groundSize = 5;

    groundV.resize(5, 3);
    groundV << 0, -1, 0,
        -groundSize, -1, -groundSize,
        -groundSize, -1, groundSize,
        groundSize, -1, groundSize,
        groundSize, -1, -groundSize;
    groundF.resize(4, 3);
    groundF << 0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 1;

}

void loadScene()
{
    for (RigidBodyInstance* rbi : bodies_)
        delete rbi;
    for (RigidBodyTemplate* rbt : templates_)
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
        std::string meshname;
        ifs >> meshname;
        meshname = prefix + std::string("meshes/") + meshname;
        double scale;
        ifs >> scale;
        RigidBodyTemplate* rbt = new RigidBodyTemplate(meshname, scale);
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
        RigidBodyInstance* rbi = new RigidBodyInstance(*rbt, c, theta, cvel, w, rho);
        templates_.push_back(rbt);
        bodies_.push_back(rbi);
    }

    // bird mesh    
    std::string birdname = prefix + std::string("meshes/bird2.obj");
    delete birdTemplate_;
    birdTemplate_ = new RigidBodyTemplate(birdname, 0.1);
}

void initSimulation()
{
    time_ = 0;
    loadScene();
    updateRenderGeometry();
}


void simulateOneStep()
{
    time_ += params_.timeStep;

    // TODO: Gather DOFs, compute forces, integrate time, write DOFs back to rigid bodies

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
        ImGui::Checkbox("Penalty Forces Enabled", &params_.penaltyEnabled);
        ImGui::InputDouble("Penalty Stiffness", &params_.penaltyStiffness);
        ImGui::InputDouble("Coefficient of Restitution", &params_.coefficientOfRestitution);
    }    

    if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Space)))
    {
        launch_ = true;
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

  sceneFile_ = "box.scn";
  birdTemplate_ = NULL;
  launch_ = false;

  initSimulation();

  polyscope::init();

  polyscope::state::userCallback = callback;

  while (!polyscope::render::engine->windowRequestsClose())
  {
      if (running_)
          simulateOneStep();
      updateRenderGeometry();

      if (launch_)
      {
          double launchVel = 100;
          Eigen::Vector3d launchPos;
          for (int i = 0; i < 3; i++)
              launchPos[i] = polyscope::view::getCameraWorldPosition()[i];
          
          Eigen::Vector3d launchDir;
          glm::vec3 look;
          glm::vec3 dummy;
          polyscope::view::getCameraFrame(look, dummy, dummy);
          for (int i = 0; i < 3; i++)
              launchDir[i] = look[i];
// TODO: launch a bird
          launch_ = false;
      }

      auto * surf = polyscope::registerSurfaceMesh("Bodies", renderQ, renderF);
      surf->setTransparency(0.9);      
      polyscope::registerSurfaceMesh("Ground", groundV, groundF);

      polyscope::frameTick();
  }

  return 0;
}

