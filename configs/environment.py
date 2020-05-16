import os
import sys

import torch

IN_COLAB = False

DRIVE_FOLDER = '/content/gdrive/My Drive/' \
        'NLP2 Xlingual'

DEVICE = 'cpu'


def init_environment():
    setup_ipython()

    setup_colab()
    check_gpu()


def setup_colab():
    global IN_COLAB

    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False

    print('IN_COLAB:', IN_COLAB)

    if IN_COLAB is True:
        sys.path.insert(0, os.getcwd())

        from google.colab import drive
        drive.mount('/content/gdrive/')
        os.chdir(DRIVE_FOLDER)


def remount_gdrive():
    from google.colab import drive
    os.chdir('/content/cross-lingual-transfer-learning/')
    drive.flush_and_unmount()
    setup_colab()


def setup_ipython():
    try:
        from IPython import get_ipython
    except:
        return

    ipython = get_ipython()
    if ipython is None:
        return

    # ipython.core.extensions.load_extension('autoreload')
    autoreload_loaded = ipython.extension_manager.load_extension('autoreload')
    if autoreload_loaded is not None and autoreload_loaded != 'already loaded':
        print('Unable to load autoreload IPython extension:',
            autoreload_loaded)
    ipython.magic('autoreload 2')


def check_gpu():
    global DEVICE

    if torch.cuda.is_available():
        DEVICE = "gpu"
    else:
        DEVICE = "cpu"
