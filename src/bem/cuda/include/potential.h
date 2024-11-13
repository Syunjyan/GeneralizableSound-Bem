#pragma once
#include "common.h"
#include "helper_math.h"

BEM_NAMESPACE_BEGIN

enum PotentialType
{
    SINGLE_LAYER,
    DOUBLE_LAYER,
    ADJOINT_DOUBLE_LAYER,
    HYPER_SINGULAR_LAYER,
    BM_LHS,
    BM_RHS
};
#define EPS 1e-3

template <PotentialType Type>
HOST_DEVICE inline complex potential(float3 x, float3 y, float3 xn, float3 yn, float k)
{
    float r = length(x - y);
    if (r < EPS)
        r = EPS;
    complex ikr = complex(0, 1) * r * k;
    if constexpr (Type == SINGLE_LAYER)
    {
        return exp(ikr) / (4 * M_PI * r);
    }
    else if constexpr (Type == DOUBLE_LAYER)
    {
        return -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(y - x, yn);
    }
    else if constexpr (Type == ADJOINT_DOUBLE_LAYER)
    {
        return -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    }
    else if constexpr (Type == HYPER_SINGULAR_LAYER)
    {
        return exp(ikr) / (4 * M_PI * r * r * r) *
               ((3 - 3 * ikr - k * k * r * r) * dot(y - x, yn) * dot(x - y, xn) / (r * r) + (1 - ikr) * dot(xn, yn));
    }
    else if constexpr (Type == BM_LHS)
    {
        return (-complex(0, 1) * k * dot(xn, yn) + 1.0 / (r * r) * (1 - ikr) * dot(y - x, yn)) * exp(ikr) /
               (4 * M_PI * r);
    }
    else if constexpr (Type == BM_RHS)
    {
        auto sp = exp(ikr) / (4 * M_PI * r);
        return -sp + complex(0, 1) / k * sp / (r * r) * (1 - ikr) * dot(x - y, xn);
    }
}

// for fast evaluation of the lhs of the boundary integral equation (repeated exp calculations is avoided)
HOST_DEVICE inline void
bm_lhs_potential(float3 x, float3 y, float3 xn, float3 yn, float k, complex &ret1, complex &ret2)
{
    float r = length(x - y);
    if (r < EPS)
        r = EPS;
    complex ikr = complex(0, 1) * r * k;
    auto sp = exp(ikr) / (4 * M_PI * r);
    ret1 = (-complex(0, 1) * k * dot(xn, yn) + 1.0 / (r * r) * (1 - ikr) * dot(y - x, yn)) * sp;
    ret2 = complex(0, 1) / k * sp;
}

BEM_NAMESPACE_END