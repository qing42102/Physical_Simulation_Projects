# Physical Simulation Projects
4 projects developed for a physical simulation class. 

## Universe of Goo Part I
Simulation for multiple mass-spring systems, which can act like a blob of goo when there are enough particles and springs. To simulate the dynamics, there is the spring force, gravitational force, and floor contact force. Implements different ODE solvers, such as explicit Euler, implicit Euler, velocity verlet, implicit midpoint, and Runge-Kutta 45. In addition, the simulation contains saws that can delete the parts of the mass-spring system that comes into contact with the saws. 

## Universe of Goo Part II
Builds upon part I by adding rigid rods and flexible rods to the simulation. Particles can be connected by either springs, rigid rods, or flexible rods. Implements different methods for handling the constraints of rigid rods, such as penalty force, step and project, and constrained Lagrangian. 

## Furious Birds Part I
Simulates the rotation and translation of a rigid body. The idea is to translate and rotate the rigid body from a template position to the current position. 

## Furious Birds Part II
The simulation allows the user to shoot a bird at blocks and then the collisions between these rigid bodies are simulated. To detect collisions, the simulation first uses a bounding box for broad-phase detection. Then the simulation checks whether any vertex of one body overlaps with another body using signed distance function. If two bodies are in collision, a penalty force is applied so that the two bodies are repelled and they are no longer in collision. 

## Cloth Encounter
Simulates cloth using the position-based dynamics method. The method replaces forces with constraints by projecting the position and velocity onto the constraints. Implements a pin force, strectching force, bending force, and pull force. 
