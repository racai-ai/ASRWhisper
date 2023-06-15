import torch
from torch import nn
import whisper

from pytorch_lightning import LightningModule

import evaluate

from transformers import (
    #AdamW,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from collator import WhisperDataCollatorWhithPadding
from config import Config

class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name, lang, train_dataset, eval_dataset) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True, beam_size=32) #beam_size = 32 or comment out
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_test_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()


        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_data=self.tokenizer.decode(l, skip_special_tokens=True)
            #print("l_data=",l_data)
            l_list.append(l_data)
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def validation_step(self, batch, batch_id):
        ret=self.validation_test_step(batch,batch_id)
        self.log("val/loss", ret["loss"], on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", ret["cer"], on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", ret["wer"], on_step=True, prog_bar=True, logger=True)

        return ret

    def test_step(self, batch, batch_id):
        #ret=self.validation_test_step(batch,batch_id)
        labels = batch["labels"].long()
        labels[labels == -100] = self.tokenizer.eot

        ret=self.model.decode(batch["input_ids"],self.options)

        #l_list=self.tokenizer.decode(batch["labels"], skip_special_tokens=True)

        o_list, l_list = [], []
        for i in range(0,len(ret)):
            o_list.append(ret[i].text)

        for l in labels:
            l_data=self.tokenizer.decode(l, skip_special_tokens=True)
            #print("l_data=",l_data)
            l_list.append(l_data)


        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)


        #self.log("test/loss", ret["loss"], on_step=True, prog_bar=True, logger=True)
        #self.log("test/cer", ret["cer"], on_step=True, prog_bar=True, logger=True)
        #self.log("test/wer", ret["wer"], on_step=True, prog_bar=True, logger=True)
        self.log("test/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("test/wer", wer, on_step=True, prog_bar=True, logger=True)

        return ret

    def predict_step(self, batch, batch_id):
        #print(batch["input_ids"].shape[0])
        #sys.exit(-1)
        ret=self.model.decode(batch["input_ids"],self.options)
        for i in range(0,len(ret)):
            batch["dataset"].setText(batch["dataIndex"][i],ret[i].text)
        return ret

    def configure_optimizers(self):
        print("configure_optimizers")
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        print("setup")
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.__train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.__eval_dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
