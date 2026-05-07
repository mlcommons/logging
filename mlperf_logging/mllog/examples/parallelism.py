# Copyright 2019 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================

import os

from mlperf_logging import mllog


def parallelism_example():
  """Example usage of mllog with parallelism and config keys"""

  mllogger = mllog.get_mllogger()

  mllog.config(
      filename="parallelism_example.log",
      default_namespace="worker1",
      default_stack_offset=1,
      default_clear_line=False,
      root_dir=os.path.normpath(
          os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))

  mllogger.start(key=mllog.constants.RUN_START)

  # Log the model config file used for this run
  mllogger.event(key=mllog.constants.CONFIG_FILENAME, value="llama31_405b_config.yaml")

  # Log parallelism strategy
  mllogger.event(key=mllog.constants.TENSOR_PARALLELISM, value=8)
  mllogger.event(key=mllog.constants.PIPELINE_PARALLELISM, value=4)
  mllogger.event(key=mllog.constants.CONTEXT_PARALLELISM, value=2)
  mllogger.event(key=mllog.constants.EXPERT_PARALLELISM, value=1)

  # Log micro batch size alongside global batch size
  mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=2048)
  mllogger.event(key=mllog.constants.MICRO_BATCH_SIZE, value=1)

  mllogger.end(key=mllog.constants.RUN_STOP, metadata={mllog.constants.STATUS: mllog.constants.SUCCESS})


if __name__ == "__main__":
  parallelism_example()
