import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertTokenizer,
    BertForMaskedLM,
    get_linear_schedule_with_warmup,
)
import cross_lingual.utils as utils
from cross_lingual.datasets.utils import get_dataloader

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
CACHE_DIR = Path("cache")
LOG_DIR = Path("logs")
BEST_MODEL_FNAME = "best-model.pt"


class Trainer():
    """ Trainer instantiates the model, loads the data and sets up the training-evaluation pipeline """

    def __init__(self, config: Dict):
        """ Initialize the trainer with data, models and optimizers

        Parameters
        ---
        config:
            dictionary of configurations with the following keys:
            {
                'exp_name': "test_experiment",
                'epochs': 10,
                'batch_size': 64,
                'valid_freq': 50, 
                'save_freq': 100,
                'device': 'cpu',
                'data_dir': 'data/mtl-dataset/',
                'max_grad_norm': 1.0,
                #TODO: add more as required
            }
        """
        self.config = config

        self.data_dir = Path(config['data_dir'])
        self.checkpoint_dir = CHECKPOINTS / config['exp_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.exp_dir = RESULTS / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        utils.init_logging(log_path=self.log_dir)
        logging.info(f'Config :\n{config}')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f'Using device: {self.device}')

        self.model = BertForMaskedLM.from_pretrained(config['bert_arch'], cache_dir=CACHE_DIR).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_arch'], cache_dir=CACHE_DIR)
        self.train_dl = get_dataloader(self.data_dir / "train.txt", self.tokenizer, config['batch_size'])
        self.valid_dl = get_dataloader(self.data_dir / "valid.txt", self.tokenizer, config['batch_size'])

        self.model.resize_token_embeddings(len(self.tokenizer))
        # TODO: implement layer freezing
        # freeze_layers(self.model, n_layers=config['layers_to_freeze'])

        # apply weight decay to all parameters except bias and layer normalization
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        total_steps = len(self.train_dl) * config['epochs']
        opt = AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=config['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=config['warmup_steps'],
                                                    num_training_steps=total_steps)
        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_perplexity = 0.

    def run(self):
            """ Run the train-eval loop
            
            If the loop is interrupted manually, finalization will still be executed
            """
            try:
                print(f"Begin training for {self.config['epochs']} epochs")
                self.train()
            except KeyboardInterrupt:
                print("Manual interruption registered. Please wait to finalize...")
                self.save_checkpoint()
    
    def train(self):
        """ Main training loop """
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            for i, batch in enumerate(self.train_dl):
                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)
                
                self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                print(f"EPOCH:{epoch} STEP:{i}\t Accuracy: {results['accuracy']:.3f} Loss: {results['loss']:.3f}")

                if i % self.config['valid_freq'] == 0:
                    self.validate()
                if i % self.config['save_freq'] == 0:
                    self.save_checkpoint()
    
    def validate(self):
        """ Main validation loop """
        losses = []
        accuracies = []

        print("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):
                results = self._batch_iteration(batch, training=False)
                self.writer.add_scalar('Accuracy/Valid', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Valid', results['loss'], self.current_iter)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
            
        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        report = (f"[Validation]\t"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"Total Loss: {np.mean(losses):.3f}")
        print(report)

    def test(self):
        """ Main testing loop """
        if 'test_checkpoint' in self.config:
            self.load_checkpoint(self.config['test_checkpoint'])
        else:
            sys.exit("No test_checkpoint found in config. Must include checkpoint for testing.")

        losses = []
        accuracies = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
            
        report = (f"[Test]\t"
                  f"Accuracy: {np.mean(accuracies):.3f} "
                  f"Total Loss: {np.mean(losses):.3f}")
        return report

    def _batch_iteration(self, batch: tuple, training: bool):
        """ Iterate over one batch """

        # send tensors to model device
        x = batch[0].to(self.config['device'])
        label = batch[1].to(self.config['device'])

        if training:
            self.opt.zero_grad()
            pred = self.model(x)
            loss =  self.criterion(pred, label)
            loss.backward()
            self.opt.step()
        else:
            with torch.no_grad():
                pred = self.model(x)
                loss =  self.criterion(pred, label)

        acc = (pred.argmax(dim=1) == label).float().mean().item()
        results = {'accuracy': acc, 'loss': loss.item()}
        return results

    def save_checkpoint(self, file_name: str = None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint directory and checkpoint file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = self.checkpoint_dir / file_name
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, file_name)
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, file_name: str):
        """Load the checkpoint with the given file name

        Checkpoint must contain:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
            - optimizer state

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        """
        try:
            file_name = self.checkpoint_dir / file_name
            self.logger.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config['device'])

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")