#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
// #include <cuda_runtime.h>

namespace py = pybind11;
using namespace std;

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Unaligned;


const double DEGREE2RADIAN = M_PI / 180.0;

cv::Mat convertPano2Perspective(
    cv::Mat &im_pano,
    cv::Mat_<double> &K,
    cv::Mat_<double> &rotation,
    cv::Size &img_size);

/*
 Create instrinsic parameter
 */
cv::Mat_<double> get_intrinsic_param(cv::Size &size_per)
{
    double fov = 90.0 * DEGREE2RADIAN;
    double focal_length = (double)size_per.width/(2. * tan(fov / 2.));
    cv::Mat_<double> K = (
        cv::Mat_<double>(3, 3) << 
        focal_length, 0., (double)size_per.width/2.,
        0., focal_length, (double)size_per.height/2.,
        0., 0., 1.);
    return K;
}

/*
 Get rotation matrix from the direction given
 - Input: an array of double (3 rotation angles)
 - Output: rotation matrix
 */
cv::Mat_<double> angle2RotMat(array<double, 3> &rot_angle)
{
	cv::Mat_<double> Th2c_key = (cv::Mat_<double>(3,3) << 
		1., 0., 0.,
		0., cos(rot_angle[0]), sin(rot_angle[0]),
		0., -sin(rot_angle[0]), cos(rot_angle[0])) 
		*(cv::Mat_<double>(3,3) <<
		cos(rot_angle[1]), 0., -sin(rot_angle[1]),
		0., 1., 0.,
		sin(rot_angle[1]), 0., cos(rot_angle[1]))
		*(cv::Mat_<double>(3,3) <<
        cos(rot_angle[2]), sin(rot_angle[2]), 	0.,
		-sin(rot_angle[2]), cos(rot_angle[2]), 	0.,
		0., 0., 1.);
    return Th2c_key;
}

/*
 Get x, y pixel location of panorama image 
 from theta (rotation around y-axis) and 
 phi (rotation around x-axis)
 */
void convertThetaPhiToPanoImgXY(
	const cv::Size size,
	const double &theta,
	const double &phi,
	double &x,
	double &y)
{
    // z-axis = front direction in the panorama image
	x = (theta + M_PI) * (size.width/(2.0*M_PI)) - .5;
	y = (phi + M_PI/2) * (size.height/(M_PI)) - .5;
}

/*
 Binlinear Interpolation
 */
void interpBilinear(
	const cv::Mat &src,
	const double &u,
	const double &v,
	cv::Vec3b &pixel)
{
	pixel[0] = src.at<cv::Vec3b>((int)v, (int)u)[0]*(1.0 - (v-(int)v))*(1.0 - (u-(int)u))
		+ src.at<cv::Vec3b>((int)v+1, (int)u)[0]*((v-(int)v))*(1.0 - (u-(int)u))
		+ src.at<cv::Vec3b>((int)v, (int)u+1)[0]*(1.0 - (v-(int)v))*((u-(int)u))
		+ src.at<cv::Vec3b>((int)v+1, (int)u+1)[0]*((v-(int)v))*((u-(int)u));
	pixel[1] = src.at<cv::Vec3b>((int)v, (int)u)[1]*(1.0 - (v-(int)v))*(1.0 - (u-(int)u))
		+ src.at<cv::Vec3b>((int)v+1, (int)u)[1]*((v-(int)v))*(1.0 - (u-(int)u))
		+ src.at<cv::Vec3b>((int)v, (int)u+1)[1]*(1.0 - (v-(int)v))*((u-(int)u))
		+ src.at<cv::Vec3b>((int)v+1, (int)u+1)[1]*((v-(int)v))*((u-(int)u));
	pixel[2] = src.at<cv::Vec3b>((int)v, (int)u)[2]*(1.0 - (v-(int)v))*(1.0 - (u-(int)u))
		+ src.at<cv::Vec3b>((int)v+1, (int)u)[2]*((v-(int)v))*(1.0 - (u-(int)u))
		+ src.at<cv::Vec3b>((int)v, (int)u+1)[2]*(1.0 - (v-(int)v))*((u-(int)u))
		+ src.at<cv::Vec3b>((int)v+1, (int)u+1)[2]*((v-(int)v))*((u-(int)u));
}



cv::Mat get_image(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src,
    array<double, 3> &angles)
{
    cv::Mat rgb[3];
    cv::Mat im_pano(src[0].rows(), src[0].cols(), CV_8UC3);
    for (int i=0; i<src.size(); i++)
    {
        cv::Mat channel(src[i].rows(), src[i].cols(), CV_8UC1, src[i].data());
        rgb[i] = channel;
    }
    cv::merge(rgb, 3, im_pano);
    cv::cuda::GpuMat d_pano {im_pano.rows, im_pano.cols, CV_8UC3};
    d_pano.upload(im_pano);

    // Get intrinsic parameter:
    const int height = 360; // (int)im_pano.rows/4;
    const int width = 640; // (int)im_pano.cols/4;
    cv::Size size_per = cv::Size(width, height);
    cv::Mat_<double> K = get_intrinsic_param(size_per);
    //cout << "Intrinsic Param: " << K << endl;

    cv::Mat_<double> rotation = angle2RotMat(angles); // make it front for now
    //cout << "Rotation Matrix: " << rotation << endl;

    // Create perspective image
    cv::Mat im_perspective {width, height, CV_8UC3);
    convertPano2Perspective(im_pano, im_perspective, K, rotation, size_per);
    
    return im_pano;
}

PYBIND11_MODULE(extension, m)
{
    m.def("get_image", &get_image);

    // Cuffer protocol for return value
    py::class_<cv::Mat>(m, "Image", py::buffer_protocol())
        .def_buffer([](cv::Mat& im) -> py::buffer_info{
            return py::buffer_info(
                // pointer to buffer
                im.data,
                //size of one scalar
                sizeof(unsigned char),
                // Python struct-style format descriptor
                py::format_descriptor<unsigned char>::format(),
                // Number of dimensions
                3,
                // Buffer dimensions
                { im.rows, im.cols, im.channels() },
                // Strides (in bytes) for each index
                {
                    sizeof(unsigned char) * im.channels() * im.cols,
                    sizeof(unsigned char) * im.channels(),
                    sizeof(unsigned char)
                }
            );
        });
}