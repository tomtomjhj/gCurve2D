// Source:
// https://doc.cgal.org/latest/Optimal_transportation_reconstruction_2/Optimal_transportation_reconstruction_2_2otr2_simplest_example_8cpp-example.html
// https://doc.cgal.org/latest/Optimal_transportation_reconstruction_2/Optimal_transportation_reconstruction_2_2otr2_list_output_example_8cpp-example.html
//
// Simplest example for Optimal_transportation_reconstruction_2, with no mass
// attributes for the input points and no Wasserstein tolerance
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Optimal_transportation_reconstruction_2.h>
#include <CGAL/Real_timer.h>
#include <CGAL/point_generators_2.h>
#include <fstream>
#include <iostream>
#include <vector>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;
typedef K::Segment_2 Segment;
typedef CGAL::Optimal_transportation_reconstruction_2<K> Otr;

void run(int test_points) {
  // Generate a set of random points on the boundary of a square.
  std::vector<Point> points;
  points.reserve(test_points);

  // insert using point generator
  CGAL::Random_points_on_square_2<Point> point_generator(1.);
  std::copy_n(point_generator, test_points, std::back_inserter(points));

  CGAL::Real_timer timer{};
  timer.start();

  Otr otr(points);
  otr.set_verbose(1);
  timer.stop();
  double constr_time = timer.time();
  std::cout << "Construction time: " << constr_time << " seconds" << std::endl;
  timer.reset();
  timer.start();

  if (otr.run_until(4)) // 50 steps
    std::cerr << "All done." << std::endl;
  else
    std::cerr << "Premature ending." << std::endl;

  timer.stop();
  double otr_time = timer.time();
  std::cout << "Otr run time:      " << timer.time() << " seconds" << std::endl;
  std::cout << "Total time:        " << constr_time + otr_time << " seconds"
            << std::endl;

  // list output
  // https://doc.cgal.org/latest/Optimal_transportation_reconstruction_2/classCGAL_1_1Optimal__transportation__reconstruction__2.html#ad94fc673df480c8b2076ae542fdd0e04
  std::vector<Point> isolated_points;
  std::vector<Segment> segments;

  otr.list_output(std::back_inserter(isolated_points),
                  std::back_inserter(segments));

  std::cout << "Isolated points: " << isolated_points.size()
            << ", Edges: " << segments.size() << std::endl;
  std::cout << std::endl;

  bool output_data = false;
  if (output_data) {

    // Point: https://doc.cgal.org/latest/Kernel_23/classCGAL_1_1Point__2.html
    std::vector<Point>::iterator pit;
    for (pit = isolated_points.begin(); pit != isolated_points.end(); pit++)
      std::cout << *pit << std::endl;

    std::cout << std::endl;

    // Segment:
    // https://doc.cgal.org/latest/Kernel_23/classCGAL_1_1Segment__2.html
    std::vector<Segment>::iterator sit;
    for (sit = segments.begin(); sit != segments.end(); sit++)
      std::cout << *sit << std::endl;
    std::cout << std::endl;
  }
}

int main() {
  int num_points[] = {50, 100, 500, 1000, 5000, 10000, 50000};
  for (int i = 0; i < sizeof num_points / sizeof(int); i++) {
    std::cout << num_points[i] << " points:" << std::endl;
    run(num_points[i]);
  }

  return 0;
}