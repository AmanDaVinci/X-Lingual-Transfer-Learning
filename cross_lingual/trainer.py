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
from cross_lingual.datasets.utils import mask_tokens, get_dataloader

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
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        total_steps = len(self.train_dl) * config['epochs']
        self.opt = AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=config['adam_epsilon'])
        self.scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=config['warmup_steps'],
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
                logging.info(f"Begin training for {self.config['epochs']} epochs")
                self.train()
            except KeyboardInterrupt:
                logging.info("Manual interruption registered. Please wait to finalize...")
                self.save_checkpoint()
    
    def train(self):
        """ Main training loop """
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            for i, batch in enumerate(self.train_dl):
                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)
                
                self.writer.add_scalar('Perplexity/Train', results['perplexity'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                logging.info(f"EPOCH:{epoch} STEP:{i}\t Perplexity: {results['perplexity']:.3f} Loss: {results['loss']:.3f}")

                if i % self.config['valid_freq'] == 0:
                    self.validate()
                if i % self.config['save_freq'] == 0:
                    self.save_checkpoint()
    
    def validate(self):
        """ Main validation loop """
        losses = []
        perplexities = []

        logging.info("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):
                results = self._batch_iteration(batch, training=False)
                self.writer.add_scalar('Perplexity/Valid', results['perplexity'], self.current_iter)
                self.writer.add_scalar('Loss/Valid', results['loss'], self.current_iter)
                losses.append(results['loss'])
                perplexities.append(results['perplexity'])
            
        mean_perplexity = np.exp(np.mean(losses))
        if mean_perplexity > self.best_perplexity:
            self.best_perplexity = mean_perplexity
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        report = (f"[Validation]\t"
                  f"Perpelexity: {mean_perplexity:.3f} "
                  f"Total Loss: {np.mean(losses):.3f}")
        logging.info(report)

    def test(self, test_file: str = 'test.txt'):
        """ Main testing loop """
        if 'test_checkpoint' in self.config:
            self.load_checkpoint(self.config['test_checkpoint'])
        else:
            self.load_checkpoint(BEST_MODEL_FNAME)

        losses = []
        perplexities = []
        test_dl = get_dataloader(self.data_dir / test_file, self.tokenizer, self.config['batch_size']*2)
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                perplexities.append(results['perplexity'])
            
        report = (f"[Test]\t"
                  f"Perplexity: {np.exp(np.mean(losses)):.3f} "
                  f"Total Loss: {np.mean(losses):.3f}")
        logging.info(report)
        return report

    def _batch_iteration(self, batch: tuple, training: bool):
        """ Iterate over one batch """

        # send tensors to model device
        inputs, labels = mask_tokens(batch, self.tokenizer)
        inputs = inputs.to(self.config['device'])
        labels = labels.to(self.config['device'])

        if training:
            self.opt.zero_grad()
            outputs = self.model(inputs, masked_lm_labels=labels)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.config['max_grad_norm'])
            self.opt.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()

        else:
            with torch.no_grad():
                outputs = self.model(inputs, masked_lm_labels=labels)
                loss = outputs[0]

        perplexity = torch.exp(loss)
        results = {'perplexity': perplexity.item(), 'loss': loss.item()}
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
            'best_perplexity': self.best_perplexity,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, file_name)
        logging.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, file_name: str):
        """Load the checkpoint with the given file name

        Checkpoint must contain:
            - current epoch
            - current iteration
            - model state
            - best perplexity achieved so far
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
            self.best_perplexity = checkpoint['best_perplexity']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        except OSError:
            logging.error(f"No checkpoint exists @ {self.checkpoint_dir}")