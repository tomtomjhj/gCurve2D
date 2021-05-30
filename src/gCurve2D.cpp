#include <iomanip>

#include "gDel2D/CpuCurve.h"
#include "gDel2D/GpuCurve.h"
#include "gDel2D/GpuDelaunay.h"
#include "gDel2D/PerfTimer.h"

#include "gDel2D/CPU/PredWrapper.h"

#include "DelaunayChecker.h"
#include "InputCreator.h"

int main(int argc, char *argv[]) {

  GCurve2DInput g_input;
  GCurve2DOutput g_output;
  GpuCurve gpuCurve;

  InputCreator g_creator;
  g_creator.makePoints(1000000, DiskDistribution, g_input.pointVec, 76213898);
  gpuCurve.compute(g_input, &g_output);

  std::cout << "done1" << std::endl;

  Point2Vec c_input;
  CGAL::Random_points_in_disc_2<CPoint, Creator> c_creator(1.0);
  std::copy_n(c_creator, 1000000, std::back_inserter(c_input));
  CpuCurve_compute(c_input);

  std::cout << "done2" << std::endl;
}
