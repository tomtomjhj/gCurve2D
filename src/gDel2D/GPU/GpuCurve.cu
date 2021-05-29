#include "../GpuCurve.h"
#include "../GpuDelaunay.h"

#include <iomanip>
#include <iostream>

#include "KerCommon.h"
#include "KerDivision.h"
#include "KerPredicates.h"
#include "ThrustWrapper.h"

#define GRID_STRIDE_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

// NOTE: nvcc segfaults if these are included
// #include <CGAL/Delaunay_triangulation_2.h>
// #include <CGAL/Voronoi_diagram_2.h>

__device__ double square(double x) { return x * x; }
__device__ double determinant(double a00, double a01, double a10, double a11) {
  return a00 * a11 - a10 * a01;
}

__device__ Point2 circumcenter(double coord[3][2]) {
  double dqx = coord[1][0] - coord[0][0];
  double drx = coord[2][0] - coord[0][0];
  double dqy = coord[1][1] - coord[0][1];
  double dry = coord[2][1] - coord[0][1];

  double r2 = square(drx) + square(dry);
  double q2 = square(dqx) + square(dqy);
  double den = 2 * determinant(dqx, dqy, drx, dry);

  double dcx = determinant(dry, dqy, r2, q2) / den;
  double dcy = -determinant(drx, dqx, r2, q2) / den;

  return Point2{{dcx + coord[0][0], dcy + coord[0][1]}};
}

__global__ void DT2VDVertices(KerPoint2Array points, KerTriArray input,
                              Point2 *output) {
  GRID_STRIDE_LOOP(index, input._num) {
    const Tri tri = input._arr[index];
    double coord[3][2];
    for (int i = 0; i < 3; i++) {
      assert(tri._v[i] < points._num);
      const Point2 point = points._arr[tri._v[i]];
      coord[i][0] = point._p[0];
      coord[i][1] = point._p[1];
    }
    output[index] = circumcenter(coord);
  }
}

struct isGoodTri {
  int infId;
  __host__ __device__ bool operator()(const Tri tri) {
    for (int i = 0; i < 3; i++) {
      if (tri._v[i] >= infId)
        return false;
    }
    return true;
  }
};

void extractCrust(int s_range, const TriHVec &input, SegmentHVec &output) {
  for (auto it = input.begin(); it != input.end(); it++) {
    const Tri tri = *it;
    if (tri._v[0] < s_range && tri._v[1] < s_range)
      output.push_back(Segment{tri._v[0], tri._v[1]});
    if (tri._v[1] < s_range && tri._v[2] < s_range)
      output.push_back(Segment{tri._v[1], tri._v[2]});
    if (tri._v[2] < s_range && tri._v[0] < s_range)
      output.push_back(Segment{tri._v[2], tri._v[0]});
  }
}

void GpuCurve::compute(const GCurve2DInput &input, GCurve2DOutput *output) {

  // Let S be a finite set of points in the plane.
  // move input from CPU to GPU
  _s_points.copyFromHost(input.pointVec);

  // Let V be the vertices of the Voronoi diagram of S.
  // Compute DT
  // CPU input → GPU output
  GDel2DInputGPU dt1Input{GDel2DInput{}, &_s_points};
  GDel2DOutputGPU dt1Output;
  _v_gDel.computeGPU(dt1Input, &dt1Output);

  // filter out trash triangles
  TriDVec goodTris;
  goodTris.resize(dt1Output.triVec.size());
  auto it = thrust::copy_if(dt1Output.triVec.begin(), dt1Output.triVec.end(),
                  goodTris.begin(),
                  isGoodTri{static_cast<int>(_s_points.size())});
  goodTris.shrink(it - goodTris.begin());

  // convert to VD: compute circumcenter of triangles in GPU
  _v_points.resize(goodTris.size());
  DT2VDVertices<<<1, 1>>>(toKernelArray(_s_points), toKernelArray(goodTris),
                          toKernelPtr(_v_points));
  CudaCheckError();

  // Let D be the Delaunay triangulation of S∪V.
  _sv_points.copyFrom2(_s_points, _v_points);
  GDel2DInputGPU dt2Input{GDel2DInput{}, &_sv_points};
  GDel2DOutputGPU dt2Output;
  _sv_gDel.computeGPU(dt2Input, &dt2Output);

  // An edge of D belongs to the crust of S if both its endpoints belong to S
  // movo to cpu and extract crust
  TriHVec suv_tris;
  dt2Output.triVec.copyToHost(suv_tris);
  extractCrust(_s_points.size(), suv_tris, output->segmentVec);
}
