#include "../CpuCurve.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Voronoi_diagram_2.h>

SegmentVec CpuCurve_compute(const Point2Vec &input) {

  // Let V be the vertices of the Voronoi diagram of S.
  clock_t time;
  time = clock();
  std::unordered_set<CPoint> s_points(input.begin(), input.end());
  std::cout << ((float)(clock() - time)) / CLOCKS_PER_SEC
            << " seconds \t s_points constructor" << std::endl;
  time = clock();
  VD v_VD{s_points.begin(), s_points.end()};
  std::cout << ((float)(clock() - time)) / CLOCKS_PER_SEC
            << " seconds \t VD constructor" << std::endl;

  // Let D be the Delaunay triangulation of S∪V.
  Point2Vec sv_points;
  time = clock();
  sv_points.insert(sv_points.end(), s_points.begin(), s_points.end());
  for (auto i = v_VD.vertices_begin(); i != v_VD.vertices_end(); i++) {
    sv_points.push_back(i->point());
  }
  std::cout << ((float)(clock() - time)) / CLOCKS_PER_SEC
            << " seconds \t sv_points insert" << std::endl;
  time = clock();
  DT sv_DT{sv_points.begin(), sv_points.end()};
  std::cout << ((float)(clock() - time)) / CLOCKS_PER_SEC
            << " seconds \t DT constructor" << std::endl;

  // An edge of D belongs to the crust of S if both its endpoints belong to S
  time = clock();
  SegmentVec crust_segements;
  for (auto i = sv_DT.all_edges_begin(); i != sv_DT.all_edges_end(); i++) {
    auto face = i->first;
    auto index = i->second;
    auto source = face->vertex(CGAL::Triangulation_cw_ccw_2::cw(index));
    auto target = face->vertex(CGAL::Triangulation_cw_ccw_2::ccw(index));
    bool source_in_S = s_points.find(source->point()) != s_points.end();
    bool target_in_S = s_points.find(target->point()) != s_points.end();
    if (source_in_S && target_in_S)
      crust_segements.push_back(
          std::make_pair(source->point(), target->point()));
  }
  std::cout << ((float)(clock() - time)) / CLOCKS_PER_SEC << " seconds \t crust"
            << std::endl;
  time = clock();
  auto x = std::move(crust_segements);
  std::cout << ((float)(clock() - time)) / CLOCKS_PER_SEC
            << " seconds \t move(crust_segments)" << std::endl;
  return x;
}
