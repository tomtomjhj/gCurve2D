// #define WINDOWS_VISUALIZATION

#include <iomanip>

#include "../gDel2D/GpuDelaunay.h"
#include "../gDel2D/PerfTimer.h"

#include "../gDel2D/CPU/PredWrapper.h"

#include "../DelaunayChecker.h"
#include "../InputCreator.h"

// For testing
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Voronoi_diagram_2.h>

using namespace CGAL;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;
typedef Creator_uniform_2<double, Point> Creator;
typedef std::vector<Point> Vector;

typedef CGAL::Delaunay_triangulation_2<K> DT;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT> AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
typedef CGAL::Voronoi_diagram_2<DT, AT, AP> VD;
///////////////////////////////////////////////////////////////////////////////

double square(double x) { return x * x; }
double determinant(double a00, double a01, double a10, double a11) {
  return a00 * a11 - a10 * a01;
}

Point2 circumcenter(double coord[3][2]) {
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

std::vector<Point2> DTOutputToVDVertex(GDel2DInput const input,
                                       GDel2DOutput const output) {
  std::vector<Point2> vec{};

  auto dtVertices = input.pointVec; // vector<Point2>
  auto triIndices = output.triVec;  // vector<Tri>

  for (auto it = triIndices.begin(); it != triIndices.end(); it++) {
    double coord[3][2];
    for (int i = 0; i < 3; i++) {
      auto point = dtVertices[it.base()->_v[i]];
      coord[i][0] = point._p[0];
      coord[i][1] = point._p[1];
    }
    Point2 cc = circumcenter(coord);
    vec.push_back(cc);
  }

  return vec;
}

class App {
public:
  // Parameters
  Distribution _dist;
  int _seed;
  bool _doCheck; // Check Euler, orientation, etc.
  bool _inFile;  // Input from file
  std::string _inFilename;

  // In-Out Data
  GDel2DInput _input;
  GDel2DOutput _output;

  // Main
  int _runNum;
  int _pointNum;
  int _constraintNum;

  // Statistics
  Statistics statSum;

public:
  App() {
    _pointNum = 10;
    _constraintNum = -1;
    _dist = UniformDistribution;
    _seed = 76213898;
    _inFile = false;
    _doCheck = true;

    _runNum = 1;

    return;
  }

  void reset() {
    Point2HVec().swap(_input.pointVec);
    SegmentHVec().swap(_input.constraintVec);
    TriHVec().swap(_output.triVec);
    TriOppHVec().swap(_output.triOppVec);

    cudaDeviceReset();

    return;
  }

  void run() {
    // Pick the best CUDA device
    const int deviceIdx = cutGetMaxGflopsDeviceId();
    CudaSafeCall(cudaSetDevice(deviceIdx));
    CudaSafeCall(cudaDeviceReset());

    GpuDel gpuDel;

    for (int i = 0; i < _runNum; ++i) {
      reset();

      std::cout << "Point set: " << _seed << std::endl;

      // 1. Create points
      InputCreator creator;

      if (_inFile) {
        creator.readPoints(_inFilename, _input.pointVec, _input.constraintVec,
                           _constraintNum);

        _pointNum = _input.pointVec.size();
      } else
        creator.makePoints(_pointNum, _dist, _input.pointVec, _seed);

      // 2. Compute Delaunay triangulation
      gpuDel.compute(_input, &_output);

      // Generate DT circumcenters
      std::cout << std::endl;
      auto vdVertices = DTOutputToVDVertex(_input, _output);
      for (auto it = vdVertices.begin(); it != vdVertices.end(); it++) {
        std::cout << it->_p[0] << " " << it->_p[1] << std::endl;
      }
      std::cout << std::endl;

      // Compare with CGAL VD
      Vector cgalPoints;
      for (auto it = _input.pointVec.begin(); it != _input.pointVec.end();
           it++) {
        cgalPoints.push_back(Point(it.base()->_p[0], it.base()->_p[1]));
      }
      VD vd(cgalPoints.begin(), cgalPoints.end());
      for (auto it = vd.vertices_begin(); it != vd.vertices_end(); it++) {
        std::cout << it->point() << std::endl;
      }

      double f[3][2] = {{3, 2}, {1, 4}, {5, 4}};
      Point2 p = circumcenter(f);
      std::cout << p._p[0] << " " << p._p[1] << std::endl;

      ++_seed;
    }
  }
};

void parseCommandline(int argc, char *argv[], App &app) {
  int idx = 1;

  while (idx < argc) {
    if (0 == std::string("-n").compare(argv[idx])) {
      app._pointNum = atoi(argv[idx + 1]);
      ++idx;
    } else if (0 == std::string("-c").compare(argv[idx])) {
      app._constraintNum = atoi(argv[idx + 1]);
      ++idx;
    } else if (0 == std::string("-r").compare(argv[idx])) {
      app._runNum = atoi(argv[idx + 1]);
      ++idx;
    } else if (0 == std::string("-seed").compare(argv[idx])) {
      app._seed = atoi(argv[idx + 1]);
      ++idx;
    } else if (0 == std::string("-check").compare(argv[idx])) {
      app._doCheck = true;
    } else if (0 == std::string("-d").compare(argv[idx])) {
      const int distVal = atoi(argv[idx + 1]);
      app._dist = (Distribution)distVal;

      ++idx;
    } else if (0 == std::string("-inFile").compare(argv[idx])) {
      app._inFile = true;
      app._inFilename = std::string(argv[idx + 1]);

      ++idx;
    } else if (0 == std::string("-insAll").compare(argv[idx])) {
      app._input.insAll = true;
    } else if (0 == std::string("-noSort").compare(argv[idx])) {
      app._input.noSort = true;
    } else if (0 == std::string("-noReorder").compare(argv[idx])) {
      app._input.noReorder = true;
    }
#ifdef WINDOWS_VISUALIZATION
    else if (0 == std::string("-noViz").compare(argv[idx])) {
      Visualizer::instance()->disable();
    }
#endif
    else if (0 == std::string("-profiling").compare(argv[idx])) {
      app._input.profLevel = ProfDetail;
    } else if (0 == std::string("-diag").compare(argv[idx])) {
      app._input.profLevel = ProfDiag;
    } else if (0 == std::string("-debug").compare(argv[idx])) {
      app._input.profLevel = ProfDebug;
    } else {
      std::cout << "Error in input argument: " << argv[idx] << "\n\n";
      std::cout << "Syntax: GDelFlipping [-n PointNum][-r RunNum][-seed "
                   "SeedNum][-d DistNum]\n";
      std::cout << "                     [-inFile "
                   "FileName][-insAll][-noSort][-noReorder][-check]\n";
      std::cout << "                     [-profiling][-diag][-debug]\n";
      std::cout << "Dist: 0-Uniform 1-Gaussian 2-Disk 3-Circle 4-Grid "
                   "5-Ellipse 6-TwoLines\n";

      exit(1);
    }

    ++idx;
  }
}

int main(int argc, char *argv[]) {
  App app;

  parseCommandline(argc, argv, app);

  // Run test
  app.run();

  return 0;
}
