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

struct pair {
  double a, b;
};

pair run(int num_points, int seed) {
  GCurve2DInput g_input;
  GCurve2DOutput g_output;
  GpuCurve gpuCurve;

  InputCreator g_creator;
  clock_t gtime1, gtime2, ctime1, ctime2;

  bool custom_input = false;
  if (!custom_input) {
    g_creator.makePoints(num_points, DiskDistribution, g_input.pointVec, seed);
    gtime1 = clock();
  } else {
    std::ifstream fin;
    fin.open("../src/data/olympics_5000.xy");
    if (!fin) { // file couldn't be opened
      std::cerr << "Error: file could not be opened" << std::endl;
      exit(1);
    }
    do {
      double x, y;
      fin >> x >> y;
      g_input.pointVec.push_back(Point2{{x, y}});
    } while (!fin.eof());
  }

  gpuCurve.compute(g_input, &g_output);
  gtime2 = clock();

  // Print segment index to cout
  // for (auto it = g_output.segmentVec.begin(); it !=
  // g_output.segmentVec.end();
  //      it++) {
  //   std::cout << it.base()->_v[0] << " " << it.base()->_v[1] << std::endl;
  // }

  std::cout << "done1" << std::endl;
  std::cout << ((float)(gtime2 - gtime1)) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
  double g_return_time = ((float)(gtime2 - gtime1)) / CLOCKS_PER_SEC;

  Point2Vec c_input;
  // CGAL::Random_points_in_disc_2<CPoint, Creator> c_creator(1.0);
  // std::copy_n(c_creator, num_points, std::back_inserter(c_input));

  // Use same input as g_input
  for (auto it = g_input.pointVec.begin(); it != g_input.pointVec.end(); it++) {
    c_input.push_back(CPoint(it.base()->_p[0], it.base()->_p[1]));
  }

  // std::ifstream fin;
  // fin.open("../src/data/olympics.xy");
  // if (!fin) { // file couldn't be opened
  //   std::cerr << "Error: file could not be opened" << std::endl;
  //   exit(1);
  // }
  // do {
  //   double x, y;
  //   fin >> x >> y;
  //   c_input.push_back(CPoint(x, y));
  // } while (!fin.eof());

  ctime1 = clock();
  auto c_output = CpuCurve_compute(c_input);
  ctime2 = clock();

  // for (auto it = c_output.begin(); it != c_output.end(); it++) {
  //   std::cout << it->first << " " << it->second << std::endl;
  // }

  std::cout << "done2" << std::endl;
  std::cout << ((float)(ctime2 - ctime1)) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
  double c_return_time = ((float)(ctime2 - ctime1)) / CLOCKS_PER_SEC;

  return pair{g_return_time, c_return_time};
}

int main(int argc, char *argv[]) {

  // int num_points[] = {10000, 100000, 1000000};
  // int seeds[] = {1, 5, 107, 1005, 3171};
  // for (int j = 0; j < sizeof num_points / sizeof(int); j++) {
  //   double g_cnt = 0, c_cnt = 0;
  //   for (int i = 0; i < 30; i++) {
  //     // std::cout << num_points[i] << " points:" << std::endl;
  //     pair x = run(num_points[j], 100 * i);
  //     g_cnt += x.a;
  //     c_cnt += x.b;
  //   }
  //   std::cerr << num_points[j] << " points: " << std::endl;
  //   std::cerr << "GPU: " << g_cnt / 30 << " seconds" << std::endl;
  //   std::cerr << "CPU: " << c_cnt / 30 << " seconds" << std::endl;
  //   std::cerr << std::endl;
  // }

  run(1000000, 10);
  return 0;
}
