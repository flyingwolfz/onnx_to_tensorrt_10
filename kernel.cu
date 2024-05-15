#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "device_launch_parameters.h"





// phase to image(0-1)
__device__ inline float toPhase(float phase) {
	phase = phase + 3.141592;
	if (phase > 2.0 * 3.141592) {
		phase = phase - 2.0 * 3.141592;
	}
	if (phase < 0) {
		phase = phase + 2.0 * 3.141592;
	}
	return phase / 2.0 / 3.141592;
}


__global__ void networkOutputToPhaseKernel(float* networkOutput, int W, int H, int C) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= W || v >= H)
		return;

	int imageIndex = (u + v * W) * C;

	networkOutput[(0 * H + v) * W + u] = toPhase(networkOutput[(0 * H + v) * W + u]);
	networkOutput[(1 * H + v) * W + u] = toPhase(networkOutput[(1 * H + v) * W + u]);
	networkOutput[(2 * H + v) * W + u] = toPhase(networkOutput[(2 * H + v) * W + u]);

}






extern "C"  void networkOutputToPhase(float* networkOutput, int W, int H, int C) {
	dim3 block(24, 24);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	networkOutputToPhaseKernel << <grid, block >> > (networkOutput, W, H, C);
}