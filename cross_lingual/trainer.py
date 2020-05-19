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
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    get_linear_schedule_with_warmup,
)
import cross_lingual.utils as utils
from cross_lingual.datasets.utils import mask_tokens, get_dataloader

from configs import environment

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
CACHE_DIR = Path("cache")
LOG_DIR = Path("logs")
HIDDEN_STATE_DIR = Path("hidden-states")
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
        self.hidden_state_dir = self.exp_dir / HIDDEN_STATE_DIR
        self.hidden_state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        utils.init_logging(log_path=self.log_dir)
        logging.info(f'Config :\n{config}')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f'Using device: {self.device}')

        bert_config = BertConfig.from_pretrained(config['bert_arch'],
                                                 output_hidden_states=True,
                                                 output_attentions=True)
        self.model = BertForMaskedLM.from_pretrained(
            config['bert_arch'],
            cache_dir=CACHE_DIR,
            config=bert_config).to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(config['bert_arch'], cache_dir=CACHE_DIR)
        self.train_dl = get_dataloader(self.data_dir / "train.txt", self.tokenizer, config['batch_size'])
        self.valid_dl = get_dataloader(self.data_dir / "valid.txt", self.tokenizer, config['batch_size']*2, random_sampler=False)
        self.xnli_dl = get_dataloader(self.data_dir / "xnli.txt", self.tokenizer, config['batch_size']*2, random_sampler=False)

        self.model.resize_token_embeddings(len(self.tokenizer))

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
            self.freeze_layers()

            for i, batch in enumerate(self.train_dl):
                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)
                
                self.writer.add_scalar('Perplexity/Train', results['perplexity'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                logging.info(f"EPOCH:{epoch} STEP:{i}\t Perplexity: {results['perplexity']:.3f} Loss: {results['loss']:.3f}")

                if i % self.config['valid_freq'] == 0:
                    self.validate('valid')
                    self.validate('xnli')

                if i % self.config['save_freq'] == 0:
                    self.save_checkpoint()

                    logging.info(f"Starting copy {i}")
                    environment.copy_drive_files_remotely(
                        self.checkpoint_dir, self.exp_dir)
                    environment.delete_synced_files(
                        self.checkpoint_dir, self.hidden_state_dir)
    
    def validate(self, tag: str="valid"):
        """ Main validation loop 
        
        Parameters
        ---
        tag: string
            run validation on either "valid" or "xnli"
        """
        dl = self.valid_dl if tag == "valid" else self.xnli_dl
        losses = []
        perplexities = []

        logging.info(f"Begin evaluation over {tag} set")
        with torch.no_grad():
            for i, batch in enumerate(dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                perplexities.append(results['perplexity'])
                if i> 0: break
            
        mean_loss = np.mean(losses)
        mean_perplexity = np.exp(mean_loss)
        # save best model based on validation perplexity only
        if tag == "valid" and mean_perplexity > self.best_perplexity:
            self.best_perplexity = mean_perplexity
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        self.writer.add_scalar(f'Perplexity/{tag}', mean_perplexity, self.current_iter)
        self.writer.add_scalar(f'Loss/{tag}', mean_loss, self.current_iter)
        report = (f"[{tag}]\t"
                  f"Perpelexity: {mean_perplexity:.3f} "
                  f"Loss: {mean_loss:3f}")
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
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if training:
            # don't we need self.model.train()
            # dropout layers are currently disabled

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

                hidden_states = outputs[2]
                attentions = outputs[3]
                self.save_hidden_states(hidden_states, attentions)

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
            logging.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config['device'])

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_perplexity = checkpoint['best_perplexity']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        except OSError:
            logging.error(f"No checkpoint exists @ {self.checkpoint_dir}")

    def freeze_layers(self):
        freeze_gradually = self.config.get('freeze_gradually', False)
        freeze_static_layers = self.config.get('freeze_static_layers', 0)

        assert not (bool(freeze_gradually) and bool(freeze_static_layers)), \
            'Cannot have both gradual freezing and static freezing'

        # start by un-freezing everything
        # (not the optimal way, but makes the latter computations easier)
        for param in self.model.bert.parameters():
            param.requires_grad = True

        # always freeze embedding layer
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        if freeze_gradually is True:
            total_steps = len(self.train_dl) * self.config['epochs']
            current_iter = self.current_iter + len(self.train_dl)
            unfrozen_layers = int(np.floor(current_iter / total_steps *
                (len(self.model.bert.encoder.layer))))

            max_frozen_layer = max(
                0, len(self.model.bert.encoder.layer) - unfrozen_layers)
            bert_layers_to_freeze = range(0, max_frozen_layer)
        elif freeze_static_layers:
            assert (0 <= freeze_static_layers) and \
                (freeze_static_layers <= len(self.model.bert.encoder.layer)), \
                'You can not freeze more layers than the present amount'

            max_frozen_layer = freeze_static_layers
        else:
            max_frozen_layer = 0

        bert_layers_to_freeze = range(0, max_frozen_layer)

        for layer_idx in bert_layers_to_freeze:
            for param in self.model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False


    def save_hidden_states(self, hidden_states, attentions):
        fn = f'Epoch[{self.current_epoch}]-Step[{self.current_iter}].npy'
        fn = self.hidden_state_dir / fn

        with open(fn, 'ab') as f:
            for i in range(1, len(self.model.bert.encoder.layer)+1):
                layer_hidden_state = hidden_states[i].detach().cpu().numpy()
                np.save(f, layer_hidden_state)

            for i in range(0, len(self.model.bert.encoder.layer)):
                layer_attention = attentions[i].detach().cpu().numpy()
                np.save(f, layer_attention)
