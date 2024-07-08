// main.cu
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "nbody.cuh"
#include <iostream>
#include <cstdlib>  // For rand()

const int numParticles = 100;
Particle* particles;
GLuint vbo;
struct cudaGraphicsResource* vbo_res;

void initParticles(Particle* p, int n) {
    for (int i = 0; i < n; ++i) {
        p[i].pos.x = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].pos.y = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].vel.x = (rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.01f; // Small random velocity
        p[i].vel.y = (rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.01f; // Small random velocity
        p[i].size = 0.1f;
    }
}

void updateAndRender() {
    glClear(GL_COLOR_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    Particle* d_particles;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &vbo_res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, vbo_res);

    float dt = 0.01f;

    dim3 threadsPerBlock(256); // Example: 256 threads per block
    dim3 blocksPerGrid((numParticles + threadsPerBlock.x - 1) / threadsPerBlock.x);

    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, dt);

    cudaGraphicsUnmapResources(1, &vbo_res, 0);

    // Setup vertex attribute pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, pos));
    glEnableVertexAttribArray(0);

    // Attribute for particle size (assuming size is stored as a float in Particle)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, size));
    glEnableVertexAttribArray(1);

    // Set the point size directly here
    glPointSize(5.0f); // Example: Set a larger point size for all particles

    // Render particles
    glDrawArrays(GL_POINTS, 0, numParticles);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA N-Body Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Allocate and initialize particles
    particles = (Particle*)malloc(numParticles * sizeof(Particle));
    initParticles(particles, numParticles);

    // Create vertex buffer object (VBO)
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(Particle), particles, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&vbo_res, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        updateAndRender();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(vbo_res);
    glDeleteBuffers(1, &vbo);
    free(particles);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}


