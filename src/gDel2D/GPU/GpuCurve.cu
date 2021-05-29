#include "../GpuCurve.h"

#include <iomanip>
#include <iostream>

#include "KerCommon.h"
#include "KerDivision.h"
#include "KerPredicates.h"
#include "ThrustWrapper.h"

// NOTE: nvcc segfaults if these are included
// #include <CGAL/Delaunay_triangulation_2.h>
// #include <CGAL/Voronoi_diagram_2.h>

void GpuCurve::compute(const GCurve2DInput &input, GCurve2DOutput *output) {

  // Let S be a finite set of points in the plane.
  GDel2DInput gDelInput;
  gDelInput.pointVec = input.pointVec;
  GDel2DOutput gDelOutput;


  // Let V be the vertices of the Voronoi diagram of S.
  // Compute DT and convert to VD: compute circumcenter of triangles
  gDel.compute(gDelInput, &gDelOutput);
  // TODO

  // TODO: compute S∪V
  // Let D be the Delaunay triangulation of S∪V.

  // An edge of D belongs to the crust of S if both its endpoints belong to S

}
