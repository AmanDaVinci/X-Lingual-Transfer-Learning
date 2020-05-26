import os
import re

import numpy as np

from configs import environment


layer_count = 12
state_baseline_fn = os.path.join(
    environment.DRIVE_FOLDER,
    'results/bert-dutch/hidden-states/Epoch[0]-Step[0].npy')


def read_hidden_states(fn):
    layer_states = []
    attentions = []
    with open(fn, 'rb') as f:
        for i in range(0, layer_count):
            layer_state = np.load(f)
            layer_states.append(layer_state)

        for i in range(0, layer_count):
            attention = np.load(f)
            attentions.append(attention)

    layer_states = np.stack(layer_states)
    attentions = np.stack(attentions)

    return layer_states, attentions


def compare_hidden_states(ls1, ls2, layer):
    diff = ls1[layer, :] - ls2[layer, :]
    return np.mean(np.linalg.norm(diff, ord=2, axis=-1))




def get_state_files(dir_name):
    files = os.listdir(dir_name)
    epoch_step_pairs = []

    for f in files:
        match = re.search(r'Epoch\[(\d+)]-Step\[(\d+)]\.npy', f)
        if not match:
            continue

        epoch = int(match.group(1))
        step = int(match.group(2))
        epoch_step_pairs.append((epoch, step))

    epoch_step_pairs.sort()

    return epoch_step_pairs


def get_distances(dir_name):
    layer_states_baseline, attentions_baseline = \
        read_hidden_states(state_baseline_fn)

    epoch_step_pairs = get_state_files(dir_name)

    distances = []
    for epoch, step in epoch_step_pairs:
        fn = os.path.join(dir_name, f'Epoch[{epoch:d}]-Step[{step:d}].npy')
        layer_states, attentions = read_hidden_states(fn)

        step_result = {}
        for layer in range(layer_count):
            distance = compare_hidden_states(
                layer_states_baseline, layer_states, layer)
            step_result[layer] = distance

        print(f'Epoch: {epoch:02d}, {step:06d}', step_result)
        distances.append(step_result)

    return distances
