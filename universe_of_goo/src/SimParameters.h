#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 0.001;
        integrator = TI_EXPLICIT_EULER;
        NewtonMaxIters = 20;
        NewtonTolerance = 1e-8;

        gravityEnabled = true;
        gravityG = -9.8;
        springsEnabled = true;
        springStiffness = 100;
        maxSpringStrain = 0.5;
        dampingEnabled = true;
        dampingStiffness = 1.0;
        floorEnabled = true;

        clickMode = CM_ADDPARTICLE;
        particleMass = 1.0;
        maxSpringDist = 0.25;
        particleFixed = false;

        sawRadius = 0.1;
    }

    enum ClickMode
    {
        CM_ADDPARTICLE,
        CM_ADDSAW
    };
    enum TimeIntegrator
    {
        TI_EXPLICIT_EULER,
        TI_IMPLICIT_EULER,
        TI_IMPLICIT_MIDPOINT,
        TI_VELOCITY_VERLET
    };

    double timeStep;
    TimeIntegrator integrator;
    double NewtonTolerance;
    int NewtonMaxIters;

    bool gravityEnabled;
    double gravityG;
    bool springsEnabled;
    double springStiffness;
    double maxSpringStrain;
    bool floorEnabled;
    bool dampingEnabled;
    double dampingStiffness;

    ClickMode clickMode;
    double particleMass;
    double maxSpringDist;
    bool particleFixed;
    double sawRadius;
};

extern SimParameters params_;

#endif