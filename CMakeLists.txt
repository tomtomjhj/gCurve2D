project(gCurve2d)
cmake_minimum_required(VERSION 3.12)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(CGAL REQUIRED)
set(CMAKE_BUILD_TYPE Release)
# See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_61,code=sm_61)
#add_definitions(-DREAL_TYPE_FP32)
include_directories(
    src
    )
cuda_add_executable(
    ${PROJECT_NAME}
      src/DelaunayChecker.cpp
      src/HashTable.cpp
      src/RandGen.cpp
      src/Visualizer.cpp
      src/InputCreator.cpp
      src/gDel2D/CPU/PredWrapper.cpp
      src/gDel2D/CPU/predicates.cpp
      src/Main.cpp
      src/gDel2D/GPU/GpuDelaunay.cu
      src/gDel2D/GPU/ThrustWrapper.cu
      src/gDel2D/GPU/KerPredicates.cu
      src/gDel2D/GPU/KerDivision.cu
      src/gDel2D/GPU/SmallCounters.cu
   )
set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})

