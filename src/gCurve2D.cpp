#include <iomanip>

#include "gDel2D/GpuCurve.h"
#include "gDel2D/GpuDelaunay.h"
#include "gDel2D/PerfTimer.h"

#include "gDel2D/CPU/PredWrapper.h"

#include "DelaunayChecker.h"
#include "InputCreator.h"

int main(int argc, char *argv[]) {

  GCurve2DInput input;
  GCurve2DOutput output;
  GpuCurve gpuCurve;

  InputCreator creator;
  creator.makePoints(1000, UniformDistribution, input.pointVec, 76213898);
  gpuCurve.compute(input, &output);
}
