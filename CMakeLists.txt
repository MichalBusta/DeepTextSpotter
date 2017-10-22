cmake_minimum_required(VERSION 2.6)
project(CMPTrie)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH      "${CMAKE_BINARY_DIR}/bin") 

option (BUILD_WITH_DEBUG_INFO 	"Build with debugging information" ON)
option(BUILD_SHARED_LIBS        "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" OFF )
option(BUILD_PARALLEL        	  "With OpenMP" On )
set(python_version "2" CACHE STRING "Specify which Python version to use")

set(MODULES_DIR "${PROJECT_SOURCE_DIR}")

if (MSVC)
	add_definitions( -DNOMINMAX )
	add_definitions( -D_VARIADIC_MAX=10 )
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")

#-------------------Extra libraries -----------------------------------
#extra flags for OpenMP for flann
if(NOT WIN32)
	if(BUILD_PARALLEL)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        if(NOT WIN32)
            set(EXTRA_PERF_LIBS gomp)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    		add_definitions( -DPARALLEL ) 
        endif(NOT WIN32)
    endif(OPENMP_FOUND)
    else(BUILD_PARALLEL)
    message( STATUS "ANDROID BUILD")
    endif(BUILD_PARALLEL)
endif(NOT WIN32)
#-------------------Extra libraries -----------------------------------
CMAKE_POLICY(SET CMP0012 NEW) 
# ----------------------------------------------------------------------------
# Output directories 
# ----------------------------------------------------------------------------

add_subdirectory(src)

