from torch.utils.data import Dataset
import whisper
import torch
import glob
import os
from pathlib import Path
import tqdm
import sys

import re
import unicodedata

import regex

from utils import getAudio

#from asize import asizeof

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):

    def __init__(self, tokenizer):
        self.ds=[]
        self.tokenizer=tokenizer


    def loadFromFolder(self,folder):
        print("Dataset: loadFromFolder: ",folder)
        for filename in tqdm.tqdm(glob.glob(os.path.join(folder,'audio', '*.wav'))):
            txt_fname=os.path.join(folder,'text',os.path.splitext(Path(filename).name)[0]+'.txt')
            with open(txt_fname,"r") as fin: text=fin.read()
            if len(text)==0:
                print("empty text {}".format(txt_fname))
                continue

            text1=self.clean_text(text)
            if len(text1)==0:
                print("empty text after clean {}".format(txt_fname))
                continue
            #text_t=text

            text1 = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text1)
            labels = text1[1:] + [self.tokenizer.eot]
            if(len(labels)>448):
                print("labels>448 {}".format(txt_fname))
                continue
            self.ds.append({"text":text,"audio_filename":filename})

    def loadAudioFile(self,filename):
            self.ds.append({"text":"","audio_filename":filename})


    def remove_symbols(self,s: str):
        """
        Replace any other markers, symbols, punctuations with a space, keeping diacritics
        """
        return "".join(
            " " if unicodedata.category(c)[0] in "MSP" else c
            for c in unicodedata.normalize("NFKC", s)
        )

    def clean_text(self,s: str):
        text=s.lower()
        text=self.remove_symbols(text)
        text = re.sub(
            r"\s+", " ", text
        )  # replace any successive whitespace characters with a space
        text=text.strip()
        return text

    def setText(self, index,text):
        self.ds[index]["text"]=text


    def __getitem__(self, index):
        # load audio
        audio_filename=self.ds[index]["audio_filename"]
        audio=getAudio(audio_filename,0,30)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))

        # process text
        text=self.ds[index]["text"]
        text=self.clean_text(text)
        text_clean=text

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        #print(len(labels)," ",len(text)," ",audio_filename,"[",text_clean,"]")

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            "dataIndex":index,
            "text":self.ds[index]["text"],
            "text_clean":text_clean
        }

    def __len__(self):
        return len(self.ds)
