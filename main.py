import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything
import torch
import MinkowskiEngine as ME
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
import wandb

import os
from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
from pytorch_lightning.plugins import DeepSpeedPlugin
#import smdistributed.dataparallel.torch.torch_smddp
#wandb.init(dir='./wandb_dir')
os.environ["WANDB_DIR"] = './wandb'


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    #if cfg.general.get("gpus", None) is None:
    #    cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    #cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    #cfg.general.gpus =2 
    #cfg.data.num_workers = 8
    #cfg.data.pin_memory = True
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id
    ####

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        if os.path.isfile(f"{cfg.general.save_dir}/last-epoch.ckpt"):
            cfg["trainer"][
                "resume_from_checkpoint"
            ] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    ##########
    model = InstanceSegmentation(cfg)
    #model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    #num = cfg.general.checkpoint
    #cfg.general.checkpoint = None

    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        accelerator='gpu',
        devices=cfg.general.gpus,
        callbacks = callbacks,
        default_root_dir=str(cfg.general.save_dir),
        **cfg.trainer,
        num_nodes=1,
        #plugins="deepspeed_stage_2", 
        #=DeepSpeedPlugin(stage=3),
        #strategy="deepspeed",
        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=1,
            #offload_optimizer=True,  # Enable CPU Offloading
            cpu_checkpointing=True,  # (Optional) offload activations to CPU
            #offload_optimizer=True,
            #offload_parameters=True,
            #stage=3
            load_full_weights=True
            ),
        #precision=16
    )

    runner.fit(model)


    


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
        #strategy = 'ddp_spawn'
        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=1,
            #offload_optimizer=True,  # Enable CPU Offloading
            cpu_checkpointing=True,  # (Optional) offload activations to CPU
            #offload_optimizer=True,
            #offload_parameters=True,
            #stage=3
            load_full_weights=True
            ),
    )
    runner.test(model, ckpt_path="saved/checkpoint")


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
