#include <stdlib.h>
#include <stdio.h>

__global__
void matAddKernel(float** A, float** B, float** C, int n)
{
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if(j<n)
	{ 
		for (int i = 0; i < n; ++i)
			A[i][j] = B[i][j] + C[i][j];
	}
}

__global__
void matAddKernel(float** A, float** B, float** C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<n)
	{ 
		for (int j = 0; j < n; ++j)
			A[i][j] = B[i][j] + C[i][j];
	}
}

__global__
void matAddKernel(float** A, float** B, float** C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if(i<n && j<n) 
		A[i][j] = B[i][j] + C[i][j];
}

void matAdd(float** A, float** B, float** C, int n)
{
	int size = n * n * sizeof(float);
	float **d_A, **d_B, **d_C;
	
	cudaMalloc((void ***) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void ***) &d_C, size);
	cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void ***) &d_A, size);
	
	matAddKernel<<<ceil(n/2560), 256>>>(d_A, d_B, d_C, n);
	
	cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
	// Free device memory for A, B, C
	cudaFree(d_A); cudaFree(d_B); cudaFree (d_C);
}









__global__
void matvecMultKernel(float* A, float** B, float* C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<n) 
	{
		for (int j = 0; j < ; ++j)
			A[i] += B[i][j] + C[j];
	}
}

void matvecMult(float* A, float** B, float* C, int n)
{
	int size = n * sizeof(float);
	float *d_A, **d_B, *d_C;
	
	cudaMalloc((void ***) &d_B, size*n);
	cudaMemcpy(d_B, B, size*n, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_C, size);
	cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void ***) &d_A, size);
	
	matvecMultKernel<<<ceil(n/2560), 256>>>(d_A, d_B, d_C, n);
	
	cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
	// Free device memory for A, B, C
	cudaFree(d_A); cudaFree(d_B); cudaFree (d_C);
}


int main(int argc, char const *argv[])
{
	/* code */
	return 0;
}