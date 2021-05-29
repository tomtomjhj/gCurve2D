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
  // TODO

  // Statistics
  Statistics stats;
};

struct GCurve2DInput {
  Point2HVec pointVec;
};

class GpuCurve {
private:
  // For computing V (VD of input)
  GpuDel gDel;
  const GDel2DInput *_input;
  GDel2DOutput *_output;

  // For computing DT of S∪V

public:
  // TODO: input/output type?
  void compute(const GCurve2DInput &input, GCurve2DOutput *output);
}; // class GpuCurve
