CMAKE_MINIMUM_REQUIRED(VERSION 3.15 FATAL_ERROR)

PROJECT(tensorflow_custom_ops)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

INCLUDE(FindTensorFlow)

#ADD_LINK_OPTIONS(${PYTHON_TF_LFLAGS})
#ADD_COMPILE_OPTIONS(${PYTHON_TF_CFLAGS})
ADD_COMPILE_DEFINITIONS(_GLIBCXX_USE_CXX11_ABI=0)

ADD_SUBDIRECTORY(tensorflow_zero_out)
ADD_SUBDIRECTORY(tensorflow_add_one)
ADD_SUBDIRECTORY(tensorflow_reverse_sequence2)
ADD_SUBDIRECTORY(tensorflow_complex2)
ADD_SUBDIRECTORY(tensorflow_add2)
ADD_SUBDIRECTORY(tensorflow_time_two)

ADD_CUSTOM_TARGET(${PROJECT_NAME}_pkg
  COMMAND build_pip_pkg.sh
  VERBATIM
)

set(CMAKE_CXX_FLAGS "-O2")
