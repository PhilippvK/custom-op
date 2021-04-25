include(FindPackageHandleStandardArgs)

IF(TensorFlow_FOUND AND EXISTS "${TF_LIB}" AND IS_DIRECTORY "${TF_INC}")
  # TODO: fix this
  MESSAGE(STATUS "Reuse cached information from TensorFlow")
ELSE()
  SET(Python3_FIND_VIRTUALENV FIRST)
  FIND_PACKAGE (Python3 REQUIRED)

  EXECUTE_PROCESS(
    COMMAND ${Python3_EXECUTABLE} -m pip show tensorflow
    RESULT_VARIABLE EXIT_CODE
    OUTPUT_QUIET
  )

  IF(NOT ${EXIT_CODE} EQUAL 0)
    MESSAGE(FATAL_ERROR "Missing Python3 package: tensorflow")
  ENDIF()

  EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_include(), end='')" OUTPUT_VARIABLE PYTHON_TF_INC)
  EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_lib(), end='')" OUTPUT_VARIABLE PYTHON_TF_LIB)
  EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print('\\n'.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE PYTHON_TF_LFLAGS)
  EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print('\\n'.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE PYTHON_TF_CFLAGS)

  FIND_PATH(TensorFlow_INCLUDE_DIR
    NAMES
    tensorflow/core/framework/op_kernel.h
    HINTS
    ${PYTHON_TF_INC}
  )

  FIND_LIBRARY(TensorFlow_LIBRARY
    NAMES tensorflow_framework.2
    HINTS
    ${PYTHON_TF_LIB}
  )

  # set TensorFlow_FOUND
  find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

  # set external variables for usage in CMakeLists.txt
  IF(TensorFlow_FOUND)
    SET(TF_LIB ${TensorFlow_LIBRARY} CACHE PATH "TensorFlow Library")
    SET(TF_INC ${TensorFlow_INCLUDE_DIR} CACHE PATH "TensorFLow Include Dir")
  ENDIF()
ENDIF()
