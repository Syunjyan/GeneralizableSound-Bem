#pragma once
#include <common.h>

BEM_NAMESPACE_BEGIN

template <int LINE_GAUSS_NUM>
HOST_DEVICE inline void set_line_gauss_params(float *points, float *weights)
{
    // This function will be specialized for different values of LINE_GAUSS_NUM
}

// Specialization for 1-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_line_gauss_params<1>(float *points, float *weights)
{
    weights[0] = 2.000000000000000;
    points[0] = 0.000000000000000;
}

// Specialization for 2-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_line_gauss_params<2>(float *points, float *weights)
{
    weights[0] = 1.000000000000000;
    weights[1] = 1.000000000000000;
    points[0] = 0.577350269189626;
    points[1] = -0.577350269189626;
}

// Specialization for 3-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_line_gauss_params<3>(float *points, float *weights)
{
    weights[0] = 0.555555555555554;
    weights[1] = 0.888888888888889;
    weights[2] = 0.555555555555554;
    points[0] = 0.774596669241483;
    points[1] = 0.000000000000000;
    points[2] = -0.774596669241483;
}

// Specialization for 4-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_line_gauss_params<4>(float *points, float *weights)
{
    weights[0] = 0.347854845137454;
    weights[1] = 0.652145154862546;
    weights[2] = 0.652145154862546;
    weights[3] = 0.347854845137454;
    points[0] = 0.861136311594052;
    points[1] = 0.339981043584856;
    points[2] = -0.339981043584856;
    points[3] = -0.861136311594052;
}

template <int TRI_GAUSS_NUM>
HOST_DEVICE inline void set_tri_gauss_params(float *points, float *weights)
{
    // This function will be specialized for different values of TRI_GAUSS_NUM
}

// Specialization for 1-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_tri_gauss_params<1>(float *points, float *weights)
{
    weights[0] = 1.0000000000000000;
    points[0] = 0.3333333333333330;
    points[1] = 0.3333333333333330;
}

// Specialization for 3-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_tri_gauss_params<3>(float *points, float *weights)
{
    weights[0] = 0.3333333333333329;
    weights[1] = 0.3333333333333329;
    weights[2] = 0.3333333333333329;
    points[0] = 0.1666666666666670;
    points[1] = 0.1666666666666660;
    points[2] = 0.6666666666666670;
    points[3] = 0.1666666666666660;
    points[4] = 0.1666666666666670;
    points[5] = 0.6666666666666661;
}

// Specialization for 4-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_tri_gauss_params<4>(float *points, float *weights)
{
    weights[0] = -0.5625000000000000;
    weights[1] = 0.5208333333333330;
    weights[2] = 0.5208333333333330;
    weights[3] = 0.5208333333333330;
    points[0] = 0.3333333333333330;
    points[1] = 0.3333333333333340;
    points[2] = 0.2000000000000000;
    points[3] = 0.2000000000000000;
    points[4] = 0.6000000000000000;
    points[5] = 0.2000000000000000;
    points[6] = 0.2000000000000000;
    points[7] = 0.6000000000000000;
}

// Specialization for 6-point Gaussian quadrature
template <>
HOST_DEVICE inline void set_tri_gauss_params<6>(float *points, float *weights)
{
    weights[0] = 0.2233815896780110;
    weights[1] = 0.1099517436553220;
    weights[2] = 0.2233815896780110;
    weights[3] = 0.2233815896780110;
    weights[4] = 0.1099517436553220;
    weights[5] = 0.1099517436553220;
    points[0] = 0.4459484909159651;
    points[1] = 0.4459484909159651;
    points[2] = 0.0915762135097710;
    points[3] = 0.0915762135097700;
    points[4] = 0.1081030181680700;
    points[5] = 0.4459484909159651;
    points[6] = 0.4459484909159651;
    points[7] = 0.1081030181680700;
    points[8] = 0.8168475729804590;
    points[9] = 0.0915762135097700;
    points[10] = 0.0915762135097710;
    points[11] = 0.8168475729804580;
}

BEM_NAMESPACE_END