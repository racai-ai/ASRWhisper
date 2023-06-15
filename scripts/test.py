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


    model_name = "medium" #medium
    lang = "ro"
    experiment= "lr-6_5_cv_vp_rsc_rtasc_rss_corola" # cv, vox, rsc, rtasc, cv_rsc_vp_rtasc, rscsent, corola

    model_check="./content/{}/{}/artifacts/checkpoint/checkpoint-epoch=0003.ckpt".format(model_name,experiment)

    cfg = Config()

    model = WhisperModelModule(cfg, model_name, lang, [], [])

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=-1
    )

    ALL1=[
            "/data/CORPORA/SPEECH/clean/RTASC/test",
            "/data/CORPORA/SPEECH/clean/RSC/test",
            "/data/CORPORA/SPEECH/clean/CDEP-eval/test",
            "/data/CORPORA/SPEECH/clean/SSC-eval1/test",
            "/data/CORPORA/SPEECH/clean/SSC-eval2/test"
    ]

    ALL2=[
            "/data/CORPORA/SPEECH/clean/RSC/test",
            "/data/CORPORA/SPEECH/clean/RSS/test",
            "/data/CORPORA/SPEECH/clean/COROLA/test",
    ]




    testData=[
        ["/data/CORPORA/SPEECH/clean/RTASC/test"],
        ["/data/CORPORA/SPEECH/clean/RSC/test"],
        ["/data/CORPORA/SPEECH/clean/CDEP-eval/test"],
        ["/data/CORPORA/SPEECH/clean/SSC-eval1/test"],
        ["/data/CORPORA/SPEECH/clean/SSC-eval2/test"],
        ALL1,
        ["/data/CORPORA/SPEECH/clean/commonvoice/test"],
        ["/data/CORPORA/SPEECH/clean/voxpopuli/test"],
        ["/data/CORPORA/SPEECH/clean/RSS/test"],
        ["/data/CORPORA/SPEECH/clean/COROLA/test"],
        ALL2,
        ["/data/CORPORA/USPDATRO/CORPUS/processed"],
        ["/data/CORPORA/SPEECH/clean/FLEURS/test"]

    ]

    for test in testData:
        dataset_test=MyDataset(wtokenizer)
        for folder in test:
            dataset_test.loadFromFolder(folder)

        trainer.test(
            model,
            dataloaders=[torch.utils.data.DataLoader(dataset_test,
                          batch_size=cfg.batch_size,
                          num_workers=30,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )], 
            ckpt_path=model_check
        )
