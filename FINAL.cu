
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

typedef struct
{ 
    float x, y, z, vx, vy, vz; 
} Body;

__global__ void bodyForce(Body *p, float dt, int n) //Tornando a fun GPU
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Exclui o for inicial 
    
    if (i < n) 
    {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) 
        {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }
        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
}

int main(const int argc, const char** argv) 
{
    int nBodies = 2<<11;
    

    if (argc > 1)nBodies = 2<<atoi(argv[1]);

    const char * initialized_values;
    const char * solution_values;

    if (nBodies == 2<<11) 
    {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } 
    else // nBodies == 2<<15
    {
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) 
        initialized_values = argv[2];
    if (argc > 3) 
        solution_values = argv[3];

    const float dt = 0.01f;
    const int nIters = 10;

    cudaError_t bodyForceErr;
    cudaError_t asyncErr;

    int DeviceNum;
    int QuantidadeSmi;
    
    cudaGetDevice(&DeviceNum);
    cudaDeviceGetAttribute(&QuantidadeSmi, cudaDevAttrMultiProcessorCount, DeviceNum);

    int NumeroDeThreads = 128;
    int NumeroDeBlocos = 32 * QuantidadeSmi;

    float *buf;
    int bytes = nBodies*sizeof(Body);

    cudaMallocManaged(&buf, bytes);//MallocCUDA
    Body *p = (Body*)buf;

    read_values_from_file(initialized_values, buf, bytes);

    double totalTime = 0.0; 
    for (int iter = 1; iter <= nIters; iter++) 
    {
        StartTimer();

        bodyForce<<<NumeroDeBlocos, NumeroDeThreads>>>(p, dt, nBodies);

        bodyForceErr = cudaGetLastError();
        if(bodyForceErr != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(bodyForceErr));
    
        asyncErr = cudaDeviceSynchronize();
        if(asyncErr != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(asyncErr));

        for (int i = 0 ; i < nBodies; i++) 
        { // integrate position
            p[i].x += p[i].vx*dt;
            p[i].y += p[i].vy*dt;
            p[i].z += p[i].vz*dt;
        }

        const double tElapsed = GetTimer() / 1000.0;
        if (iter > 1)
            totalTime += tElapsed; 
    }
    double avgTime = totalTime / (double)(nIters-1); 

    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, buf, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);
    
    cudaFree(buf);//FREECUDA
}