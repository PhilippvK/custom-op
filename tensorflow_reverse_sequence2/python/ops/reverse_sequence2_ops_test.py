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
"""Tests for reverse_sequence2 ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_reverse_sequence2.python.ops.reverse_sequence2_ops import reverse_sequence2
except ImportError:
  from reverse_sequence2_ops import reverse_sequence2


class ReverseSequence2Test(test.TestCase):

  def testReverseSequence2(self):
    with self.test_session():
      dtype = np.int32
      len_dtype = np.int64
      seq_lengths = np.asarray([7, 2, 3, 5], dtype=len_dtype)
      inp = np.asarray([
            [1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]
          ], dtype=dtype)

      truth = np.asarray([
            [0, 0, 5, 4, 3, 2, 1, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [3, 2, 1, 4, 0, 0, 0, 0],
            [5, 4, 3, 2, 1, 6, 7, 8]
          ], dtype=dtype)

      seq_axis = 1
      batch_axis = 0
      self.assertAllClose(
          reverse_sequence2(inp, seq_lengths, batch_dim=batch_axis, seq_dim=seq_axis), truth, atol=1e-10)


if __name__ == '__main__':
  test.main()
