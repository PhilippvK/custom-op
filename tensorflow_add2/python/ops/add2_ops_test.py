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
"""Tests for add2 ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_add2.python.ops.add2_ops import add2
except ImportError:
  from add2_ops import add2


class Add2Test(test.TestCase):

  def testAdd2(self):
    with self.test_session():
      self.assertAllClose(
          add2([[1, 2], [3, 4]], [[5, 6], [7, 8]]), np.array([[6, 8], [10, 12]]))


if __name__ == '__main__':
  test.main()
