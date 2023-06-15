from pathlib import Path

import os
import numpy as np
import sys

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio
import torchaudio.transforms as at

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
import evaluate

from dataset import MyDataset

import multiprocessing

from config import Config
from whisper_module import WhisperModelModule

if __name__ == '__main__':
    multiprocessing.freeze_support()

    #BATCH_SIZE = 8 # in config.py
    TRAIN_RATE = 0.8

    SEED = 3407
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    seed_everything(SEED, workers=True)

    woptions = whisper.DecodingOptions(language="ro", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ro", task=woptions.task)


    train_name = "whisper"
    train_id = "00001"

    model_name = "medium" # large-v2
    lang = "ro"
    experiment="lr-6_5_cv_vp_rsc_rtasc_rss_corola"

    log_output_dir = "./content/{}/{}/logs".format(model_name,experiment)
    check_output_dir = "./content/{}/{}/artifacts".format(model_name,experiment)

    cfg = Config()

    os.makedirs(log_output_dir,exist_ok=True)
    os.makedirs(check_output_dir,exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    dataset_train=MyDataset(wtokenizer)
    dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/commonvoice/train")
    dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/RSC/train")
    dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/voxpopuli/train")
    dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/RTASC/train")
    dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/RSS/train")
    #dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/RSC_sentences/train")
    dataset_train.loadFromFolder("/data/CORPORA/SPEECH/clean/COROLA/train")

    #for i in range(0,len(dataset_train)):
    #    if len(dataset_train[i]["labels"])>400: print(dataset_train[i])
    #print("done")
    #sys.exit(-1)

    dataset_dev=MyDataset(wtokenizer)
    dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/commonvoice/dev")
    dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/RSC/dev")
    dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/voxpopuli/dev")
    dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/RTASC/dev")
    dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/RSS/dev")
    #dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/RSC_sentences/dev")
    dataset_dev.loadFromFolder("/data/CORPORA/SPEECH/clean/COROLA/dev")

    #for i in range(0,len(dataset_dev)):
    #    if len(dataset_dev[i]["dec_input_ids"])<10: print(dataset_dev[i])
    #print("done")
    #sys.exit(-1)

    model = WhisperModelModule(cfg, model_name, lang, dataset_train, dataset_dev)

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    trainer.fit(model)
