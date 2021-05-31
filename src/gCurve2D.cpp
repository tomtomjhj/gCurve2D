#include <ctime>
#include <fstream>
#include <iomanip>

#include "gDel2D/CpuCurve.h"
#include "gDel2D/GpuCurve.h"
#include "gDel2D/GpuDelaunay.h"
#include "gDel2D/PerfTimer.h"

#include "gDel2D/CPU/PredWrapper.h"

#include "DelaunayChecker.h"
#include "InputCreator.h"

void run(int num_points) {
  GCurve2DInput g_input;
  GCurve2DOutput g_output;
  GpuCurve gpuCurve;

  InputCreator g_creator;
  g_creator.makePoints(1000000, DiskDistribution, g_input.pointVec, 76213898);
  // std::ifstream fin;
  // fin.open("../src/data/olympics_5000.xy");
  // if (!fin) { // file couldn't be opened
  //   std::cerr << "Error: file could not be opened" << std::endl;
  //   exit(1);
  // }
  // do {
  //   double x, y;
  //   fin >> x >> y;
  //   g_input.pointVec.push_back(Point2{{x, y}});
  // } while (!fin.eof());
  // gpuCurve.compute(g_input, &g_output);

  std::cout << "done1" << std::endl;

  Point2Vec c_input;
  CGAL::Random_points_in_disc_2<CPoint, Creator> c_creator(1.0);
  std::copy_n(c_creator, 1000000, std::back_inserter(c_input));
  CpuCurve_compute(c_input);

  std::cout << "done2" << std::endl;
}

int main(int argc, char *argv[]) {
  /*
  int num_points[] = {50, 100, 500, 1000, 5000, 10000, 50000};
  for (int i = 0; i < sizeof num_points / sizeof(int); i++) {
    std::cout << num_points[i] << " points:" << std::endl;
    run(num_points[i]);
  }*/
  run(0);
  return 0;
}
