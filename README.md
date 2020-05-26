# Evolution of Representations in Cross-Lingual Fine-tuning
Cross Lingual Transfer Learning project, conducted as part of the Natural Language Processing 2 course at Universiteit van Amsterdam.
This work has been done by Aman Hussain and Emil Dudev, with Dennis Ulmer as supervisor.

## Abstract

Cross-lingual fine-tuning has been widely used to bridge the gap between high-resource \& low-resource languages. In this paper, we study the evolution of the learned representations during cross-lingual fine-tuning. We fine-tune a pre-trained multi-lingual BERT on a small Dutch corpus. A BERT model, pre-trained on Dutch exclusively, is used as a comparative baseline. We show that our transferred multi-lingual BERT learns a different representation subspace than the native model. Additionally, we explore the loss in multi-lingual capacity during fine-tuning.

## Code organisation

Our library dependencies are listedin [`./requirements.txt`](./requirements.txt).

The backbone of our code is located in [`./cross_lingual/`](./cross_lingual/), while training and testing are managed by [`./main.py`](./main.py).
Samples commands:
```
$ ./main.py --config=configs.finetune_bert --train
$ ./main.py --config=configs.finetune_bert_gradual --test
```
Configuration for the training and test processes are located in [`./configs/`](./configs/), having each config comprise a single dictionary with parameters.

During the training process, checkpoints (as well as the best trained model) are stored in a new directory, [`./checkpoints/`](./checkpoints/), while logs and hidden state values are stored in [`./results/`](./results/).

After training is (sufficiently) complete, analysis can be performed on the trained models and logged hidden states.
All our analysis code is present in [`./notebooks/`](./notebooks/).

Our core source code is organized under [`./cross_lingual/`](./cross_lingual/) as follows:
* [`datasets/`](./cross_lingual/datasets/): Dataset class for reading the Dutch and XNLI corpus
* [`trainer.py`](./cross_lingual/trainer.py): Implements the train, validationa and test routine
* [`utils.py`](./cross_lingual/utils.py): utility functions for logging experiments
* [`hidden_state_utils.py`](./cross_lingual/hidden_state_utils.py): functions for experimenting with hidden states

## Environment

All notebooks (including [`trainer.ipynb`](./notebooks/trainer.ipynb)) have been run on [Google Colab](https://colab.research.google.com/).
We do note that our saved checkpoints and activations take about 1 GiB of disk space per item. Our total disk usage for the whole experiment amounted about 2 TiB.


## Paper

[Evolution of Representations in Cross-Lingual Fine-tuning][Hussain_Dudev_xlingual]


[Hussain_Dudev_xlingual]: ./report/Evolution_of_Representations_in_Cross-Lingual_Fine-tuning.pdf
