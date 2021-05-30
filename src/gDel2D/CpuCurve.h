#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/point_generators_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 CPoint;
typedef CGAL::Creator_uniform_2<double, CPoint> Creator;
typedef std::vector<CPoint> Point2Vec;
typedef std::vector<std::pair<CPoint, CPoint>> SegmentVec;

typedef CGAL::Delaunay_triangulation_2<K> DT;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT> AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
typedef CGAL::Voronoi_diagram_2<DT, AT, AP> VD;

SegmentVec CpuCurve_compute(const Point2Vec &input);
