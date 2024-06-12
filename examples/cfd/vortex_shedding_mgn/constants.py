# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Tuple

from pydantic import BaseModel


class Constants(BaseModel):
    """vortex shedding constants"""

    # wb configs
    wandb_mode: str = "online"
    watch_model: bool = True

    # Model name
    model_name: str = "training"

    # data configs
    data_dir: str = "/datasets/cylinder_flow/cylinder_flow"

    # training configs
    epochs: int = 5  # 25
    training_batch_size: int = 11
    num_training_samples: int = 22  # 1000
    num_training_time_steps: int = 100  # 600
    training_noise_std: float = 0.02

    valid_batch_size: int = 1 # Must be 1 for now
    num_valid_samples: int = 4  # 100
    num_valid_time_steps: int = 200  # 600

    lr: float = 0.0001
    lr_decay_rate: float = 0.9999991
    ckpt_path: str = "/workspace/checkpoints_training_6"
    ckpt_name: str = "model.pt"

    # Mesh Graph Net Setup
    num_input_features: int = 6
    num_edge_features: int = 3
    num_output_features: int = 3
    processor_size: int = 15
    num_layers_node_processor: int = 2
    num_layers_edge_processor: int = 2
    hidden_dim_processor: int = 128
    hidden_dim_node_encoder: int = 128
    num_layers_node_encoder: int = 2
    hidden_dim_edge_encoder: int = 128
    num_layers_edge_encoder: int = 2
    hidden_dim_node_decoder: int = 128
    num_layers_node_decoder: int = 2
    aggregation: str = "sum"
    do_concat_trick: bool = False
    num_processor_checkpoint_segments: int = 0
    activation_fn: str = "elu"

    # performance configs
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    num_test_samples: int = 100
    num_test_time_steps: int = 600
    viz_vars: Tuple[str, ...] = ("u", "v", "p")
    frame_skip: int = 10
    frame_interval: int = 1
