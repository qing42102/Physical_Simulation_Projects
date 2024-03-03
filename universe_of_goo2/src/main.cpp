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
#include "SceneObjects.h"
#include "numerical_ODE.cpp"
#include "update_objects.cpp"

bool running_;
SimParameters params_;
double time_;
std::vector<Particle, Eigen::aligned_allocator<Particle>> particles_;
std::vector<Connector *> connectors_;
std::vector<Saw> saws_;
std::vector<BendingStencil> bendingStencils_;

Eigen::MatrixXd renderQ;
Eigen::MatrixXi renderF;
Eigen::MatrixXd renderC;

struct MouseClick
{
    double x;
    double y;
    SimParameters::ClickMode mode;
};

std::deque<MouseClick> mouseClicks_;

void initSimulation()
{
    time_ = 0;
    particles_.clear();
    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
        delete *it;
    connectors_.clear();
    saws_.clear();
    bendingStencils_.clear();
}

void updateRenderGeometry()
{
    double baseradius = 0.02;
    double pulsefactor = 0.1;
    double pulsespeed = 50.0;

    int sawteeth = 20;
    double sawdepth = 0.1;
    double sawangspeed = 10.0;

    double baselinewidth = 0.005;

    int numcirclewedges = 20;

    // this is terrible. But, easiest to get up and running

    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3d> vertexColors;
    std::vector<Eigen::Vector3i> faces;

    int idx = 0;

    double eps = 1e-4;

    if (params_.floorEnabled)
    {
        for (int i = 0; i < 6; i++)
        {
            vertexColors.push_back(Eigen::Vector3d(0.3, 1.0, 0.3));
        }

        verts.push_back(Eigen::Vector3d(-2, -0.5, eps));
        verts.push_back(Eigen::Vector3d(2, -0.5, eps));
        verts.push_back(Eigen::Vector3d(-2, -1, eps));

        faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));

        verts.push_back(Eigen::Vector3d(-2, -1, eps));
        verts.push_back(Eigen::Vector3d(2, -0.5, eps));
        verts.push_back(Eigen::Vector3d(2, -1, eps));
        faces.push_back(Eigen::Vector3i(idx + 3, idx + 4, idx + 5));
        idx += 6;
    }

    for (std::vector<Connector *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
    {
        switch ((*it)->getType())
        {
        case SimParameters::CT_SPRING:
        {
            Eigen::Vector3d color;
            if ((*it)->associatedBendingStencils.empty())
                color << 0.0, 0.0, 1.0;
            else
                color << 0.75, 0.5, 0.75;
            Eigen::Vector2d sourcepos = particles_[(*it)->p1].pos;
            Eigen::Vector2d destpos = particles_[(*it)->p2].pos;

            Eigen::Vector2d vec = destpos - sourcepos;
            Eigen::Vector2d perp(-vec[1], vec[0]);
            perp /= perp.norm();

            double dist = (sourcepos - destpos).norm();

            double width = baselinewidth / (1.0 + 20.0 * dist * dist);

            for (int i = 0; i < 4; i++)
                vertexColors.push_back(color);

            verts.push_back(Eigen::Vector3d(sourcepos[0] + width * perp[0], sourcepos[1] + width * perp[1], -eps));
            verts.push_back(Eigen::Vector3d(sourcepos[0] - width * perp[0], sourcepos[1] - width * perp[1], -eps));
            verts.push_back(Eigen::Vector3d(destpos[0] + width * perp[0], destpos[1] + width * perp[1], -eps));
            verts.push_back(Eigen::Vector3d(destpos[0] - width * perp[0], destpos[1] - width * perp[1], -eps));

            faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));
            faces.push_back(Eigen::Vector3i(idx + 2, idx + 1, idx + 3));
            idx += 4;

            break;
        }
        case SimParameters::CT_RIGIDROD:
        {
            Eigen::Vector3d color;
            if ((*it)->associatedBendingStencils.empty())
                color << 1.0, 0.0, 1.0;
            else
                color << 1.0, 1.0, 0.3;

            Eigen::Vector2d sourcepos = particles_[(*it)->p1].pos;
            Eigen::Vector2d destpos = particles_[(*it)->p2].pos;
            Eigen::Vector2d vec = destpos - sourcepos;
            Eigen::Vector2d perp(-vec[1], vec[0]);
            perp /= perp.norm();

            double width = baselinewidth;

            for (int i = 0; i < 4; i++)
                vertexColors.push_back(color);

            verts.push_back(Eigen::Vector3d(sourcepos[0] + width * perp[0], sourcepos[1] + width * perp[1], -eps));
            verts.push_back(Eigen::Vector3d(sourcepos[0] - width * perp[0], sourcepos[1] - width * perp[1], -eps));
            verts.push_back(Eigen::Vector3d(destpos[0] + width * perp[0], destpos[1] + width * perp[1], -eps));
            verts.push_back(Eigen::Vector3d(destpos[0] - width * perp[0], destpos[1] - width * perp[1], -eps));

            faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));
            faces.push_back(Eigen::Vector3i(idx + 2, idx + 1, idx + 3));
            idx += 4;

            break;
        }
        default:
            break;
        }
    }

    int nparticles = particles_.size();

    for (int i = 0; i < nparticles; i++)
    {
        double radius = baseradius * sqrt(getTotalParticleMass(i));
        radius *= (1.0 + pulsefactor * sin(pulsespeed * time_));

        Eigen::Vector3d color(0, 0, 0);

        if (particles_[i].fixed)
        {
            radius = baseradius;
            color << 1.0, 0, 0;
        }

        for (int j = 0; j < numcirclewedges + 2; j++)
        {
            vertexColors.push_back(color);
        }

        verts.push_back(Eigen::Vector3d(particles_[i].pos[0], particles_[i].pos[1], 0));

        const double PI = 3.1415926535898;
        for (int j = 0; j <= numcirclewedges; j++)
        {
            verts.push_back(Eigen::Vector3d(particles_[i].pos[0] + radius * cos(2 * PI * j / numcirclewedges),
                                            particles_[i].pos[1] + radius * sin(2 * PI * j / numcirclewedges), 0));
        }

        for (int j = 0; j <= numcirclewedges; j++)
        {
            faces.push_back(Eigen::Vector3i(idx, idx + j + 1, idx + 1 + ((j + 1) % (numcirclewedges + 1))));
        }

        idx += numcirclewedges + 2;
    }

    for (std::vector<Saw>::iterator it = saws_.begin(); it != saws_.end(); ++it)
    {
        double outerradius = it->radius;
        double innerradius = (1.0 - sawdepth) * outerradius;

        Eigen::Vector3d color(0.5, 0.5, 0.5);

        int spokes = 2 * sawteeth;
        for (int j = 0; j < spokes + 2; j++)
        {
            vertexColors.push_back(color);
        }

        verts.push_back(Eigen::Vector3d(it->pos[0], it->pos[1], 0));

        const double PI = 3.1415926535898;
        for (int i = 0; i <= spokes; i++)
        {
            double radius = (i % 2 == 0) ? innerradius : outerradius;
            verts.push_back(Eigen::Vector3d(it->pos[0] + radius * cos(2 * PI * i / spokes + sawangspeed * time_),
                                            it->pos[1] + radius * sin(2 * PI * i / spokes + sawangspeed * time_), 0));
        }

        for (int j = 0; j <= spokes; j++)
        {
            faces.push_back(Eigen::Vector3i(idx, idx + j + 1, idx + 1 + ((j + 1) % (spokes + 1))));
        }

        idx += spokes + 2;
    }

    renderQ.resize(verts.size(), 3);
    renderC.resize(vertexColors.size(), 3);
    for (int i = 0; i < verts.size(); i++)
    {
        renderQ.row(i) = verts[i];
        renderC.row(i) = vertexColors[i];
    }
    renderF.resize(faces.size(), 3);
    for (int i = 0; i < faces.size(); i++)
        renderF.row(i) = faces[i];
}

bool simulateOneStep()
{
    // Create configurational vectors
    Eigen::VectorXd q, lambda, qdot;
    buildConfiguration(q, lambda, qdot);
    // Use them for one step of time integration
    numericalIntegration(q, lambda, qdot);
    // Unpack the DOFs back into the particles for rendering
    unbuildConfiguration(q, lambda, qdot);

    // Cleanup: delete sawed objects and snapped springs
    pruneOverstrainedSprings();
    delete_objects();

    // Time advances
    time_ += params_.timeStep;
    return false;
}

void callback()
{
    ImGui::SetNextWindowSize(ImVec2(500., 0.));
    ImGui::Begin("UI", nullptr);

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
    if (ImGui::CollapsingHeader("UI Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Combo("Click Adds", (int *)&params_.clickMode, "Particles\0Saws\0\0");
        ImGui::Combo("Connector Type", (int *)&params_.connectorType, "Springs\0Rigid Rods\0Flexible Rods\0\0");
    }
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Combo("Constraint Handling", (int *)&params_.constraintHandling, "Penalty Method\0Step and Project\0Lagrange Multipliers\0\0");

        ImGui::InputDouble("Timestep", &params_.timeStep);
        ImGui::InputDouble("Newton Tolerance", &params_.NewtonTolerance);
        ImGui::InputInt("Newton Max Iters", &params_.NewtonMaxIters);
        ImGui::InputDouble("Penalty Stiffness", &params_.penaltyStiffness);
    }
    if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Gravity Enabled", &params_.gravityEnabled);
        ImGui::InputDouble("  Gravity g", &params_.gravityG);
        ImGui::Checkbox("Springs Enabled", &params_.springsEnabled);
        ImGui::InputDouble("  Max Strain", &params_.maxSpringStrain);
        ImGui::Checkbox("Damping Enabled", &params_.dampingEnabled);
        ImGui::InputDouble("  Viscosity", &params_.dampingStiffness);
        ImGui::Checkbox("Floor Enabled", &params_.floorEnabled);
        ImGui::Checkbox("Bending Enabled", &params_.bendingEnabled);
        // viewer.imgui->addWindow(Eigen::Vector2i(1000, 0), "New Objects");
    }

    if (ImGui::CollapsingHeader("New Particles", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Is Fixed", &params_.particleFixed);
        ImGui::InputDouble("Mass", &params_.particleMass);
    }

    if (ImGui::CollapsingHeader("New Saws", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Radius", &params_.sawRadius);
    }

    if (ImGui::CollapsingHeader("New Springs", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Max Spring Dist", &params_.maxSpringDist);
        ImGui::InputDouble("Base Stiffness", &params_.springStiffness);
    }

    if (ImGui::CollapsingHeader("New Rods", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputInt("Num Segments", &params_.rodSegments);
        ImGui::InputDouble("Density", &params_.rodDensity);
        ImGui::InputDouble("Stretching Stiffness", &params_.rodStretchingStiffness);
        ImGui::InputDouble("Bending Stiffness", &params_.rodBendingStiffness);
    }

    ImGuiIO &io = ImGui::GetIO();
    io.DisplayFramebufferScale = ImVec2(1, 1);
// this now only works on macs with retina displays - maybe funky on older macbook airs?
// more robust solution is documented here: https://github.com/ocornut/imgui/issues/5081
#if defined(__APPLE__)
    io.DisplayFramebufferScale = ImVec2(2, 2);
#endif

    if (io.MouseClicked[0] && !io.WantCaptureMouse)
    {
        MouseClick mc;
        glm::vec2 screenCoords{io.MousePos.x * io.DisplayFramebufferScale.x, io.MousePos.y * io.DisplayFramebufferScale.y};

        glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();

        glm::vec4 ndc{-1.0f + 2.0f * screenCoords.x / (polyscope::view::bufferWidth), 1.0f - 2.0f * screenCoords.y / (polyscope::view::bufferHeight), 0, 1};
        glm::vec4 camera = glm::inverse(proj) * ndc;
        mc.x = camera[0];
        mc.y = camera[1];
        mc.mode = params_.clickMode;
        mouseClicks_.push_back(mc);
    }

    ImGui::End();
}

int main(int argc, char **argv)
{
    polyscope::view::setWindowSize(1600, 800);
    polyscope::view::setWindowResizable(false);
    polyscope::view::style = polyscope::view::NavigateStyle::Planar;
    polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
    polyscope::options::buildGui = false;
    polyscope::options::openImGuiWindowForUserCallback = false;

    polyscope::options::autocenterStructures = false;
    polyscope::options::autoscaleStructures = false;

    initSimulation();

    polyscope::init();

    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::state::lengthScale = 1.;
    polyscope::state::boundingBox =
        std::tuple<glm::vec3, glm::vec3>{{-2., -1., -1.}, {2., 1., 1.}};

    polyscope::state::userCallback = callback;

    while (!polyscope::render::engine->windowRequestsClose())
    {
        if (running_)
            simulateOneStep();
        updateRenderGeometry();
        auto *surf = polyscope::registerSurfaceMesh("UI", renderQ, renderF);
        surf->setTransparency(0.9);
        auto *color = surf->addVertexColorQuantity("Colors", renderC);
        color->setEnabled(true);

        polyscope::frameTick();
        while (!mouseClicks_.empty())
        {
            MouseClick mc = mouseClicks_.front();
            mouseClicks_.pop_front();
            switch (mc.mode)
            {
            case SimParameters::ClickMode::CM_ADDPARTICLE:
            {
                addParticle(mc.x, mc.y);
                break;
            }
            case SimParameters::ClickMode::CM_ADDSAW:
            {
                addSaw(mc.x, mc.y);
                break;
            }
            }
        }
    }

    return 0;
}
