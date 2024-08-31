// nbody.cuh
#ifndef NBODY_H
#define NBODY_H

#include <cuda_runtime.h>
#include <vector_types.h> // Include CUDA vector types

struct Particle {
    float2 pos;
    float2 vel;
    float size;
};

__global__ void updateParticles(Particle* particles, int numParticles, float dt);

#endif // NBODY_H