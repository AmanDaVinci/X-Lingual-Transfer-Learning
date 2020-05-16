import os
import sys
import shutil

import torch

from dirsync import sync


IN_COLAB = False

DRIVE_FOLDER = '/content/gdrive/My Drive/' \
        'NLP2 Xlingual'

DEVICE = 'cpu'


def init_environment():
    setup_ipython()
    check_gpu()

    setup_colab()
    copy_drive_files_locally()



def copy_drive_files_locally():
    global IN_COLAB

    if IN_COLAB is False:
        return

    shutil.copytree(os.path.join(DRIVE_FOLDER, 'cache'), './cache/')
    shutil.rmtree('./data/')
    shutil.copytree(os.path.join(DRIVE_FOLDER, 'data'), './data/')


def copy_drive_files_remotely(*paths):
    global IN_COLAB

    if IN_COLAB is False:
        return

    for path in paths:
        print('d0', path, os.path.join(DRIVE_FOLDER, path))
        sync(path, os.path.join(DRIVE_FOLDER, path), 'sync', create=True, verbose=True)

    remount_gdrive()


def delete_synced_files(*paths):
    global IN_COLAB

    if IN_COLAB is False:
        return

    for path in paths:
        print('d1', path)
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def setup_colab():
    global IN_COLAB

    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False

    print('IN_COLAB:', IN_COLAB)

    if IN_COLAB is True:
        assert DEVICE == 'gpu', 'You perhaps want to switch to a GPU'

        sys.path.insert(0, os.getcwd())

        mount_gdrive()


def mount_gdrive():
    from google.colab import drive
    drive.mount('/content/gdrive/')
    # os.chdir(DRIVE_FOLDER)


def remount_gdrive():
    global IN_COLAB

    if IN_COLAB is False:
        return

    from google.colab import drive
    drive.flush_and_unmount()
    mount_gdrive()


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
