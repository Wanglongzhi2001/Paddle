#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from test_sparse_attention_op import get_cuda_version

import paddle
from paddle.base import core

# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0

is_sm_supported = (
    paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] == 9
) or (paddle.device.cuda.get_device_capability()[0] >= 9)


def _to_fp8_saturated(x: paddle.Tensor, float8_dtype) -> paddle.Tensor:
    # The default behavior in Paddle for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. So we saturate here manualy.
    if float8_dtype == paddle.float8_e4m3fn:
        x = x.clip(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clip(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    return x.to(float8_dtype)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11080
    or not is_sm_supported,
    "MatmulFp8 requires CUDA >= 11.8 and SM version >= 8.9",
)
class TestMatmulFp8(unittest.TestCase):
    def config(self):
        self.dtype = 'float8_e4m3fn'
        self.rtol = 0.7
        self.atol = 2.5
        self.x_shape = (32, 16)
        self.y_shape = (16, 32)

    def setUp(self):
        self.config()
        paddle.seed(2024)
        self.input_a = paddle.rand(self.x_shape).astype(self.dtype)
        self.input_b = paddle.rand(self.y_shape).astype(self.dtype)

    def get_reference_out(self):
        self.input_a_fp16 = self.input_a.astype("float16")
        self.input_b_fp16 = self.input_b.astype("float16")
        out = paddle.matmul(self.input_a_fp16, self.input_b_fp16)
        return out

    def get_op_out(self):
        out = paddle.matmul(self.input_a, self.input_b)
        return out

    def test_matmul_fp8(self):
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


if __name__ == '__main__':
    unittest.main()
