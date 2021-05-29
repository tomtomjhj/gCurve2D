project(gCurve2d)
cmake_minimum_required(VERSION 3.12)
find_package(CGAL REQUIRED)
set(CMAKE_VERBOSE_MAKEFILE "ON")
# See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_61,code=sm_61)
# silence clangd
# add_compile_options(-ferror-limit=0 --no-cuda-version-check)
#add_definitions(-DREAL_TYPE_FP32)
include_directories(
    src
    )

# cuda_add_executable(
#     gDel2D
#       src/Main.cpp
#       src/DelaunayChecker.cpp
#       src/HashTable.cpp
#       src/RandGen.cpp
#       src/Visualizer.cpp
#       src/InputCreator.cpp
#       src/gDel2D/CPU/PredWrapper.cpp
#       src/gDel2D/CPU/predicates.cpp
#       src/gDel2D/GPU/GpuDelaunay.cu
#       src/gDel2D/GPU/ThrustWrapper.cu
#       src/gDel2D/GPU/KerPredicates.cu
#       src/gDel2D/GPU/KerDivision.cu
#       src/gDel2D/GPU/SmallCounters.cu
#    )
# target_link_libraries(gDel2D ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(
    gCurve2D
      src/gCurve2D.cpp
      src/DelaunayChecker.cpp
      src/HashTable.cpp
      src/RandGen.cpp
      src/Visualizer.cpp
      src/InputCreator.cpp
      src/gDel2D/CPU/PredWrapper.cpp
      src/gDel2D/CPU/predicates.cpp
      src/gDel2D/GPU/GpuDelaunay.cu
      src/gDel2D/GPU/GpuCurve.cu
      src/gDel2D/GPU/ThrustWrapper.cu
      src/gDel2D/GPU/KerPredicates.cu
      src/gDel2D/GPU/KerDivision.cu
      src/gDel2D/GPU/SmallCounters.cu
   )
target_link_libraries(gCurve2D ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} CGAL::CGAL)

## cgal test
ADD_EXECUTABLE(test_cgal src/examples/otr2_example.cpp)
target_link_libraries(test_cgal CGAL::CGAL)

cuda_add_executable(
    test_cc
      src/examples/circumcenter.cpp
      src/DelaunayChecker.cpp
      src/HashTable.cpp
      src/RandGen.cpp
      src/Visualizer.cpp
      src/InputCreator.cpp
      src/gDel2D/CPU/PredWrapper.cpp
      src/gDel2D/CPU/predicates.cpp
      src/gDel2D/GPU/GpuDelaunay.cu
      src/gDel2D/GPU/ThrustWrapper.cu
      src/gDel2D/GPU/KerPredicates.cu
      src/gDel2D/GPU/KerDivision.cu
      src/gDel2D/GPU/SmallCounters.cu
   )
target_link_libraries(test_cc ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} CGAL::CGAL)
