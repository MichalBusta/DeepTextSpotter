set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src)

find_package(PythonLibs 2.7 REQUIRED)

find_package(NumPy REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("/usr/include/python2.7")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src)

add_definitions(-DUSE_SARRAY -DUSE_SARRAY_TRIE -DUSE_SARRAY_MAP2 -DINSTANTIATE_TEMPLATES -D NO_TCL)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(trie_py 
  "trie.cpp"
)

add_library(cmp_trie SHARED
	"trie_api.c"
)

target_link_libraries(cmp_trie  trie_py ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

set_target_properties( cmp_trie
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        PREFIX ""
    ) 

