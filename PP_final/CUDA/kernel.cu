# include <cuda.h>
# include <stdio.h>
# include <stdlib.h>
# include <assert.h>
extern "C"{
# include "kernel.h"
}

#define MAX_BRIGHTNESS 255

__global__ void calculate_G(float *G, float *G_x, float *G_y, int width, int height){
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = thisY * width + thisX;

    if (thisX != 0 && thisX != width-1 && thisY != 0 && thisY != height-1){
	G[idx] = (float)hypot(G_x[idx], G_y[idx]);
    }
    else{
	G[idx] = 1;
    }
}

__global__ void non_maximum_sup(float *nms, float *G, float *G_x, float *G_y, int width, int height){
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = thisY * width + thisX;

    if (thisX != 0 && thisX != width-1 && thisY != 0 && thisY != height-1){
       	const int nn = idx - width;
       	const int ss = idx + width;
       	const int ww = idx + 1;
       	const int ee = idx - 1;
       	const int nw = nn + 1;
       	const int ne = nn - 1;
       	const int sw = ss + 1;
       	const int se = ss - 1;
       	const float dir = (float) (fmod(atan2(G_y[idx], G_x[idx]) + M_PI, M_PI) / M_PI) * 8;

        if (((dir <= 1 || dir > 7) && G[idx] > G[ee] && G[idx] > G[ww]) || // 0 deg
            ((dir > 1 && dir <= 3) && G[idx] > G[nw] && G[idx] > G[se]) || // 45 deg
            ((dir > 3 && dir <= 5) && G[idx] > G[nn] && G[idx] > G[ss]) || // 90 deg
            ((dir > 5 && dir <= 7) && G[idx] > G[ne] && G[idx] > G[sw]))   // 135 deg
	    nms[idx] = G[idx];
	else
	    nms[idx] = 0;
    }
    else{
	nms[idx] = 1;
    }
}

__global__ void convolution(float *in, float *out, float *kernel, int nx, int ny, int kn, bool normalize){
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = thisY * nx + thisX;

    const int khalf = kn / 2;
    float min = 0.5;
    float max = 254.5;
    float pixel = 0.0;
    size_t c = 0;
    int i, j;

    assert(kn % 2 == 1);
    assert(nx > kn && ny > kn);

    if (thisX >= khalf && thisX < nx-khalf && thisY >= khalf && thisY < ny-khalf){
	pixel = c = 0;

        for (j = -khalf; j <= khalf; j++)
          for (i = -khalf; i <= khalf; i++)
            pixel += in[(thisY - j) * nx + thisX - i] * kernel[c++];

        if (normalize == true)
          pixel = MAX_BRIGHTNESS * (pixel - min) / (max - min);

        out[idx] = (float) pixel;
    }
    else{
	out[idx] = 0;
    }
}

/*
 * gaussianFilter: http://www.songho.ca/dsp/cannyedge/cannyedge.html
 * Determine the size of kernel (odd #)
 * 0.0 <= sigma < 0.5 : 3
 * 0.5 <= sigma < 1.0 : 5
 * 1.0 <= sigma < 1.5 : 7
 * 1.5 <= sigma < 2.0 : 9
 * 2.0 <= sigma < 2.5 : 11
 * 2.5 <= sigma < 3.0 : 13 ...
 * kernel size = 2 * int(2 * sigma) + 3;
 */
void gaussian_kernel(float *kernel,
		     const int n,
		     const float mean,
		     const float sigma)
{
  int i, j;
  size_t c = 0;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      kernel[c++] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
  }
}

/*
 * Links:
 * http://en.wikipedia.org/wiki/Canny_edge_detector
 * http://www.tomgibara.com/computer-vision/CannyEdgeDetector.java
 * http://fourier.eng.hmc.edu/e161/lectures/canny/node1.html
 * http://www.songho.ca/dsp/cannyedge/cannyedge.html
 *
 * Note: T1 and T2 are lower and upper thresholds.
 */
extern "C"
float * canny_edge_detection(const float    *in,
	                     const int      width,
		             const int      height,
			     const int      t1,
			     const int      t2,
			     const float    sigma)
{
  int i, j, k, nedges;
  int *edges;
  size_t t = 1;
  float *retval;
  int dataSize = width * height * sizeof(float);
  const int n = 2 * (int) (2 * sigma) + 3;
  const float mean = (float) floor(n / 2.0);
  size_t c = 0;

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

  const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  /* host memory */
  float *G = (float*)calloc(dataSize, 1);

  float *after_Gx = (float*)calloc(dataSize, 1);

  float *after_Gy = (float*)calloc(dataSize, 1);

  float *nms = (float*)calloc(dataSize, 1);

  float *out = (float*)malloc(dataSize);

  float *pixels = (float*)malloc(dataSize);

  float *kernel = (float*)malloc(n * n * sizeof(float));

  /* device memory */
  float *d_pixels;
  cudaMalloc(&d_pixels, dataSize);

  float *d_out;
  cudaMalloc(&d_out, dataSize);

  float *d_Gx;
  cudaMalloc(&d_Gx, dataSize);

  float *d_Gy;
  cudaMalloc(&d_Gy, dataSize);

  float *d_after_Gx;
  cudaMalloc(&d_after_Gx, dataSize);

  float *d_after_Gy;
  cudaMalloc(&d_after_Gy, dataSize);

  float *d_G;
  cudaMalloc(&d_G, dataSize);

  float *d_nms;
  cudaMalloc(&d_nms, dataSize);

  float *d_kernel;
  cudaMalloc(&d_kernel, n * n * sizeof(float));

  /* copy input data from host to device */
  cudaMemcpy(d_pixels, in, dataSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Gx, Gx, 9 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Gy, Gy, 9 * sizeof(float), cudaMemcpyHostToDevice);

  /* Gaussian Filter */
  gaussian_kernel(kernel, n, mean, sigma);
  cudaMemcpy(d_kernel, kernel, n * n * sizeof(float), cudaMemcpyHostToDevice);
  convolution<<<numBlocks, threadsPerBlock>>>(d_pixels, d_out, d_kernel, width, height, n, true);

  /* Sobel Filter */
  convolution<<<numBlocks, threadsPerBlock>>>(d_out, d_after_Gx, d_Gx, width, height, 3, false);
  convolution<<<numBlocks, threadsPerBlock>>>(d_out, d_after_Gy, d_Gy, width, height, 3, false);
  calculate_G<<<numBlocks, threadsPerBlock>>>(d_G, d_after_Gx, d_after_Gy, width, height);
  cudaMemcpy(G, d_G, dataSize, cudaMemcpyDeviceToHost);

  /* Non-maximum suppression */
  non_maximum_sup<<<numBlocks, threadsPerBlock>>>(d_nms, d_G, d_after_Gx, d_after_Gy, width, height);
  cudaMemcpy(nms, d_nms, dataSize, cudaMemcpyDeviceToHost);

  /* Reuse the array used as a stack, width * height / 2 elements should be enough. */
  edges = (int *) after_Gy;
  memset(out, 0, dataSize);
  memset(edges, 0, dataSize);

  /* Tracing edges with hysteresis. Non-recursive implementation. */
  for (j = 1; j < height - 1; j++) {
    for (i = 1; i < width - 1; i++) {
      /* Trace edges. */
      if (nms[t] >= t2 && out[t] == 0) {
        out[t] = MAX_BRIGHTNESS;
        nedges = 1;
        edges[0] = t;

        do {
          nedges--;
          const int e = edges[nedges];

          int nbs[8]; // neighbours
          nbs[0] = e - width;     // nn
          nbs[1] = e + width;     // ss
          nbs[2] = e + 1;      // ww
          nbs[3] = e - 1;      // ee
          nbs[4] = nbs[0] + 1; // nw
          nbs[5] = nbs[0] - 1; // ne
          nbs[6] = nbs[1] + 1; // sw
          nbs[7] = nbs[1] - 1; // se

          for (k = 0; k < 8; k++) {
            if (nms[nbs[k]] >= t1 && out[nbs[k]] == 0) {
              out[nbs[k]] = MAX_BRIGHTNESS;
              edges[nedges] = nbs[k];
              nedges++;
            }
          }
        } while (nedges > 0);
      }
      t++;
    }
  }

  retval = (float*)malloc(dataSize);

  /* Convert back to float */
  for (i = 0; i < width * height; i++) {
    retval[i] = (float)out[i];
  }

  cudaFree(d_pixels);
  cudaFree(d_out);
  cudaFree(d_Gx);
  cudaFree(d_Gy);
  cudaFree(d_after_Gx);
  cudaFree(d_after_Gy);
  cudaFree(d_G);
  cudaFree(d_nms);
  cudaFree(d_kernel);
  free(after_Gx);
  free(after_Gy);
  free(G);
  free(nms);
  free(pixels);
  free(out);
  free(kernel);

  return retval;
}
