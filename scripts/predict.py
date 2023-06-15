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
from collator import WhisperDataCollatorWhithPadding

if __name__ == '__main__':
    multiprocessing.freeze_support()

    BATCH_SIZE = 2

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    woptions = whisper.DecodingOptions(language="ro", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ro", task=woptions.task)


    model_name = "tiny"
    lang = "ro"
    experiment="cv"

    model_check="./content/{}/{}/artifacts/checkpoint/checkpoint-epoch=0009.ckpt".format(model_name,experiment)

    cfg = Config()

    dataset_predict=MyDataset(wtokenizer)
    dataset_predict.loadAudioFile("/data/CORPORA/SPEECH/clean/RTASC/train/audio/S136_5.wav")

    model = WhisperModelModule(cfg, model_name, lang, [], [])

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=-1
    )

    ret=trainer.predict(
        model,
        dataloaders=[torch.utils.data.DataLoader(dataset_predict,
                          batch_size=cfg.batch_size,
                          num_workers=0,
                          collate_fn=WhisperDataCollatorWhithPadding(keepDataIndex=True,dataset=dataset_predict)
                          )], 
        return_predictions=False,
        ckpt_path=model_check
    )

    for item in dataset_predict:
        print(item["text"])
