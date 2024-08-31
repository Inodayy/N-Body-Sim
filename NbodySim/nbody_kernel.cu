// nbody_kernel.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nbody.cuh"
#include <cmath> // For sqrtf and fabsf


__global__ void updateParticles(Particle* particles, int numParticles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        float2 force = { 0.0f, 0.0f };
        for (int j = 0; j < numParticles; ++j) {
            if (i != j) {
                float dx = particles[j].pos.x - particles[i].pos.x;
                float dy = particles[j].pos.y - particles[i].pos.y;
                float dist = sqrtf(dx * dx + dy * dy);
                float f = 0.001f / (dist * dist + 0.01f); // Simplified gravity with softening
                force.x += f * dx;
                force.y += f * dy;
            }
        }
        particles[i].vel.x += force.x * dt;
        particles[i].vel.y += force.y * dt;
        particles[i].pos.x += particles[i].vel.x * dt;
        particles[i].pos.y += particles[i].vel.y * dt;
    }
}

//// CUDA Kernel to compute forces
//__global__ void computeForces(Particle* particles, float* forcesX, float* forcesY, int numParticles) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < numParticles) {
//        float fx = 0.0f, fy = 0.0f;
//        for (int j = 0; j < numParticles; ++j) {
//            if (i != j) {
//                float dx = particles[j].x - particles[i].x;
//                float dy = particles[j].y - particles[i].y;
//                float distSqr = dx * dx + dy * dy + SOFTENING;
//                float invDist = 1.0f / sqrtf(distSqr);
//                float invDistCube = invDist * invDist * invDist;
//                float f = G * particles[i].mass * particles[j].mass * invDistCube;
//                fx += f * dx;
//                fy += f * dy;
//            }
//        }
//        forcesX[i] = fx;
//        forcesY[i] = fy;
//    }
//}