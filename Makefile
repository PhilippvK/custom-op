CXX := g++
PYTHON_BIN_PATH = python

ZERO_OUT_SRCS = $(wildcard tensorflow_zero_out/cc/kernels/*.cc) $(wildcard tensorflow_zero_out/cc/ops/*.cc)
ADD_ONE_SRCS = $(wildcard tensorflow_add_one/cc/kernels/*.cc) $(wildcard tensorflow_add_one/cc/ops/*.cc)
REVERSE_SEQUENCE_SRCS = $(wildcard tensorflow_reverse_sequence2/cc/kernels/*.cc) $(wildcard tensorflow_reverse_sequence2/cc/ops/*.cc)
COMPLEX2_SRCS = $(wildcard tensorflow_complex2/cc/kernels/*.cc) $(wildcard tensorflow_complex2/cc/ops/*.cc)
ADD2_SRCS = $(wildcard tensorflow_add2/cc/kernels/*.cc) $(wildcard tensorflow_add2/cc/ops/*.cc)
#TIME_TWO_SRCS = tensorflow_time_two/cc/kernels/time_two_kernels.cc $(wildcard tensorflow_time_two/cc/kernels/*.h) $(wildcard tensorflow_time_two/cc/ops/*.cc)
TIME_TWO_SRCS = tensorflow_time_two/cc/kernels/time_two_kernels.cc $(wildcard tensorflow_time_two/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = tensorflow_zero_out/python/ops/_zero_out_ops.so
ADD_ONE_TARGET_LIB = tensorflow_add_one/python/ops/_add_one_ops.so
REVERSE_SEQUENCE2_TARGET_LIB = tensorflow_reverse_sequence2/python/ops/_reverse_sequence2_ops.so
COMPLEX2_TARGET_LIB = tensorflow_complex2/python/ops/_complex2_ops.so
ADD2_TARGET_LIB = tensorflow_add2/python/ops/_add2_ops.so
TIME_TWO_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.so

# zero_out op
zero_out_op: $(ZERO_OUT_TARGET_LIB)

$(ZERO_OUT_TARGET_LIB): $(ZERO_OUT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

# add_one op
add_one_op: $(ADD_ONE_TARGET_LIB)

$(ADD_ONE_TARGET_LIB): $(ADD_ONE_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

add_one_test: tensorflow_add_one/python/ops/add_one_ops_test.py tensorflow_add_one/python/ops/add_one_ops.py $(ADD_ONE_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_add_one/python/ops/add_one_ops_test.py

add_one_pip_pkg: $(ADD_ONE_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

# reverse_sequence2 op
reverse_sequence2_op: $(REVERSE_SEQUENCE2_TARGET_LIB)

$(REVERSE_SEQUENCE2_TARGET_LIB): $(REVERSE_SEQUENCE2_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

reverse_sequence2_test: tensorflow_reverse_sequence2/python/ops/reverse_sequence2_ops_test.py tensorflow_reverse_sequence2/python/ops/reverse_sequence2_ops.py $(REVERSE_SEQUENCE2_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_reverse_sequence2/python/ops/reverse_sequence2_ops_test.py

reverse_sequence2_pip_pkg: $(REVERSE_SEQUENCE2_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

# complex2 op
complex2_op: $(COMPLEX2_TARGET_LIB)

$(COMPLEX2_TARGET_LIB): $(COMPLEX2_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

complex2_test: tensorflow_complex2/python/ops/complex2_ops_test.py tensorflow_complex2/python/ops/complex2_ops.py $(COMPLEX2_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_complex2/python/ops/complex2_ops_test.py

complex2_pip_pkg: $(COMPLEX2_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

# add2
add2_op: $(ADD2_TARGET_LIB)

$(ADD2_TARGET_LIB): $(ADD2_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

add2_test: tensorflow_add2/python/ops/add2_ops_test.py tensorflow_add2/python/ops/add2_ops.py $(ADD2_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_add2/python/ops/add2_ops_test.py

add2_pip_pkg: $(ADD2_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

# time_two op
time_two_op: $(TIME_TWO_TARGET_LIB)
$(TIME_TWO_TARGET_LIB): $(TIME_TWO_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

time_two_test: tensorflow_time_two/python/ops/time_two_ops_test.py tensorflow_time_two/python/ops/time_two_ops.py $(TIME_TWO_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_time_two/python/ops/time_two_ops_test.py

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(ADD_ONE_TARGET_LIB) $(REVERSE_SEQUENCE2_TARGET_LIB) $(COMPLEX2_TARGET_LIB) $(ADD2_TARGET_LIB) $(TIME_TWO_TARGET_LIB)
