# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for complex2 ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_complex2.python.ops.complex2_ops import complex2, real2, imag2
except ImportError:
  from complex2_ops import complex2, real2, imag2


class Complex2Test(test.TestCase):

  # TODO
  def testComplex2(self):
    with self.test_session():
      self.assertAllClose(
          complex2([[1, 2], [3, 4]]), np.array([[1, 0], [0, 0]]))

  def testReal2(self):
    with self.test_session():
      self.assertAllClose(
          real2([[1, 2], [3, 4]]), np.array([[1, 0], [0, 0]]))

  def testImag2(self):
    with self.test_session():
      self.assertAllClose(
          imag22([[1, 2], [3, 4]]), np.array([[1, 0], [0, 0]]))


if __name__ == '__main__':
  test.main()
