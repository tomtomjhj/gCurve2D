#pragma once

#include <iomanip>
#include <iostream>

#include "CommonTypes.h"
#include "PerfTimer.h"

#include "GPU/CudaWrapper.h"
#include "GPU/DPredWrapper.h"
#include "GPU/HostToKernel.h"
#include "GPU/SmallCounters.h"
#include "GpuDelaunay.h"

struct GCurve2DOutput {
  SegmentHVec segmentVec;
};

struct GCurve2DInput {
  Point2HVec pointVec;
};

class GpuCurve {
private:
  // input points
  Point2DVec _s_points;

  // For computing V (VD of input)
  GpuDel _v_gDel;
  TriDVec _v_tris;
  Point2DVec _v_points;

  // For computing DT of S∪V
  GpuDel _sv_gDel;
  Point2DVec _sv_points;

public:
  void compute(const GCurve2DInput &input, GCurve2DOutput *output);
}; // class GpuCurve
