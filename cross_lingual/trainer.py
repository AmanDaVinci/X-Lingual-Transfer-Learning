import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
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
                #TODO: add more as required
            }
        """
        pass

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