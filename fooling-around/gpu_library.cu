#include <sstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);                                
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call), (msg), __FILE__, __LINE__)

/*
 Main function to convert panorama image to perspective image
 */
__global__ 
void pano2perspective(
    unsigned char *pano,
    unsigned char *pers,
    double *im2ori,
    int pano_w, int pano_h,
    int pers_w, int pers_h,
    int pano_step, int pers_step)
{
    // 2D Index of current thread
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only valid threads perform memory I/O 
	if((i < pers_w) && (j < pers_h))
    {
        // Location of colored pixel in output
        int pers_tid = j * pers_step + (3 * i);
        // Create orientation matrix
        double ori[3];
        for(int m=0; m<3; m++)
        {
            ori[m] = im2ori[m*3] * (double)i
                + im2ori[m*3+1] * (double)j
                + im2ori[m*3+2]; 
        }
        
        double D = sqrt(
            ori[0] * ori[0]
            + ori[1] * ori[1]
            + ori[2] * ori[2]);
        
        double phi = asin(ori[1] / D); // [-pi/2:pi/2]
        double theta = atan2(ori[0], ori[2]); // [-pi:pi]
        double u = (theta + M_PI) * (pano_w/(2.0*M_PI)) - .5;
        double v = (phi + M_PI/2) * (pano_h/(M_PI)) - .5;
        int px0 = (int)v*pano_step + 3*(int)u;
        int px1 = (int)(v+1)*pano_step + 3*(int)u;
        int px2 = (int)v*pano_step + 3*(int)(u+1);
        int px3 = (int)(v+1)*pano_step + 3*(int)(u+1);
        double v0 = (1.0-(v-(int)v)) * (1.0-(u-(int)u));
        double v1 = ((v-(int)v))*(1.0 - (u-(int)u));
        double v2 = (1.0 - (v-(int)v))*((u-(int)u));
        double v3 = ((v-(int)v))*((u-(int)u));
        pers[pers_tid] = static_cast<unsigned char>(
            pano[px0] * v0
            + pano[px1] * v1
            + pano[px2] * v2
            + pano[px3] * v3
        );
        pers[pers_tid + 1] = static_cast<unsigned char>(
            pano[px0 + 1] * v0
            + pano[px1 + 1] * v1
            + pano[px2 + 1] * v2
            + pano[px3 + 1] * v3
        );
        pers[pers_tid + 2] = static_cast<unsigned char>(
            pano[px0 + 2] * v0
            + pano[px1 + 2] * v1
            + pano[px2 + 2] * v2
            + pano[px3 + 2] * v3
        );
    }
}

void process_image(
    const cv::Mat &pano,
    cv::Mat &pers,
    const cv::Mat_<double> &rot,
    const cv::Mat_<double> &K)
{
    // preprocess
	cv::Mat T_im2ori = rot.inv() * K.inv();
    
    // Calculate total number of bytes of input image and orientation matrix
    const int panoBytes = pano.step * pano.rows;
    const int persBytes = pers.step * pers.rows;
    const int doubleBytes = T_im2ori.step * T_im2ori.rows;

    // Return pointers
    unsigned char *d_pano, *d_pers;
    double *im2ori = new double[9];
    double *d_im2ori;
       
    // Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&d_pano, panoBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_pers, persBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc((void **)&d_im2ori, 9*sizeof(double)),"CUDA Malloc Failed");

    for(int i=0; i<3; i++)
    {
        im2ori[i*3] = T_im2ori.at<double>(i, 0);
        im2ori[i*3 + 1] = T_im2ori.at<double>(i, 1);
        im2ori[i*3 + 2] = T_im2ori.at<double>(i, 2);
    }

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_pano, pano.ptr(), panoBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_im2ori, im2ori, 9*sizeof(double), cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    
    // Specify device
    // int device = 0;
    // cudaGetDevice(&device);

    // Specify a reasonable block size
    const dim3 block(16, 16);
    //Calculate grid size to cover the whole image
    const dim3 grid((pers.cols + block.x - 1)/block.x, (pers.rows + block.y - 1)/block.y);
                    
    // Launch the color conversion kernel
    // std::cout << pano.step << " " << pano.cols << " " << pano.rows << std::endl;
    pano2perspective<<<grid, block>>>(d_pano, d_pers, d_im2ori,
                                    pano.cols, pano.rows,
                                    pers.cols, pers.rows, 
                                    pano.step, pers.step);
    
    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
    
    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(pers.ptr(), d_pers, persBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    
    //Free the device memory
    SAFE_CALL(cudaFree(d_pano),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_pers),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_im2ori),"CUDA Free Failed");

}
