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

import argparse
import os
import time
from typing import Optional, Any

import torch
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel

import wandb as wb

try:
    import apex
except:
    pass

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from modulus.distributed.manager import DistributedManager

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants import Constants


class MGNTrainer:
    def __init__(self, wb, dist, rank_zero_logger, config):
        self.dist = dist
        self.config = config

        # instantiate dataset
        rank_zero_logger.info("Loading the training dataset...")
        dataset = VortexSheddingDataset(
            name="vortex_shedding_train",
            data_dir=config.data_dir,
            split="train",
            num_samples=config.num_training_samples,
            num_steps=config.num_training_time_steps,
            noise_std=config.noise_std,
            force_reload=False,
            verbose=False,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        # instantiate validation dataset
        rank_zero_logger.info("Loading the validation dataset...")
        valid_dataset = VortexSheddingDataset(
            name="vortex_shedding_valid",
            data_dir=config.data_dir,
            split="valid",
            num_samples=config.num_valid_samples,
            num_steps=config.num_valid_time_steps,
            noise_std=config.noise_std,
            force_reload=False,
            verbose=False,
        )

        # instantiate validation dataloader
        self.valid_dataloader = GraphDataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

        # instantiate test dataset
        rank_zero_logger.info("Loading the test dataset...")
        test_dataset = VortexSheddingDataset(
            name="vortex_shedding_test",
            data_dir=config.data_dir,
            split="test",
            num_samples=config.num_test_samples,
            num_steps=config.num_test_time_steps,
            noise_std=config.noise_std,
            force_reload=False,
            verbose=False,
        )

        # instantiate test dataloader
        self.test_dataloader = GraphDataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            input_dim_nodes=config.num_input_features,
            input_dim_edges=config.num_edge_features,
            output_dim=config.num_output_features,
            processor_size=config.processor_size,
            num_layers_node_processor=config.num_layers_node_processor,
            num_layers_edge_processor=config.num_layers_edge_processor,
            hidden_dim_processor=config.hidden_dim_processor,
            hidden_dim_node_encoder=config.hidden_dim_node_encoder,
            num_layers_node_encoder=config.num_layers_node_encoder,
            hidden_dim_edge_encoder=config.hidden_dim_edge_encoder,
            num_layers_edge_encoder=config.num_layers_edge_encoder,
            hidden_dim_node_decoder=config.hidden_dim_node_decoder,
            num_layers_node_decoder=config.num_layers_node_decoder,
            aggregation=config.aggregation,
            do_concat_trick=config.do_concat_trick,
            num_processor_checkpoint_segments=config.num_processor_checkpoint_segments,
            activation_fn=config.activation_fn,
        )
        if config.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)
        if config.watch_model and not config.jit and dist.rank == 0:
            wb.watch(self.model)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=config.lr)
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: config.lr_decay_rate ** epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(config.ckpt_path, config.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def train(self, graph):
        # enable train mode
        self.model.train()

        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.config.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            loss = self.criterion(pred, graph.ndata["y"])
            return loss

    def backward(self, loss):
        # backward pass
        if self.config.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def get_lr(self):
        # get the learning rate
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    # @torch.no_grad()
    # def validation(self):
    #     # enable train mode
    #     self.model.eval()
    #     error = 0
    #     for graph in self.validation_dataloader:
    #         graph = graph.to(self.dist.device)
    #         pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
    #         gt = graph.ndata["y"]
    #         error += relative_lp_error(pred, gt)
    #     error = error / len(self.validation_dataloader)
    #     self.wb.log({"val_error (%)": error})
    #     self.rank_zero_logger.info(f"Validation error (%): {error}")


def setup_config(wandb_config):
    constant = Constants(**wandb_config)

    return constant


def main(project: Optional[str] = "modulus_gnn", entity: Optional[str] = "limitingfactor", **kwargs: Any):
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    if project is None:
        project = "modulus_gnn"

    if project is None:
        entity = "limitingfactor"

    # initialize loggers
    run = initialize_wandb(
        project=project,
        entity=entity,
        mode="online"
    )  # Wandb logger

    C = setup_config(wandb_config=run.config)
    if kwargs["activation_fn"]:
        C.activation_fn = kwargs["activation_fn"]

    # save constants to JSON file
    if dist.rank == 0:
        os.makedirs(C.ckpt_path, exist_ok=True)
        with open(
                os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
        ) as json_file:
            json_file.write(C.json(indent=4))

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    trainer = MGNTrainer(wb, dist, rank_zero_logger, C)
    start = time.time()
    rank_zero_logger.info("Training started...")
    for epoch in range(trainer.epoch_init, C.epochs):
        loss_agg = 0
        for graph in trainer.dataloader:
            loss = trainer.train(graph)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, time per epoch: {(time.time() - start):10.3e}"
        )
        wb.log({"loss_train": loss_agg})

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", "-e", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default=None)
    parser.add_argument("--activation_fn", type=str, default=None)
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    opts = get_options()

    main(**opts)
