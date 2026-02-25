"""
***Serial Position Experiments***

Serial key-value associative retrieval task
Assess performance for items at different positions in the list
"""

import sys, random, uuid, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import pipeline

if 'google.colab' in sys.modules:
    os.system('git clone https://github.com/sflippl/models-of-memory.git')
    sys.path.append('models-of-memory')
    dir = 'models-of-memory'
else:
    print("Running locally")
    dir = '.'


# note: make sure {dir}/_data/wasnorm_wordpool.txt is available if using word stimuli
def generate_stimuli(list_length=100, type='uuids'):
    if 'uuid' in type:
        return [str(uuid.uuid4()) for _ in range(list_length*2)]
    elif 'word' in type:
        with open(f'{dir}/_data/wasnorm_wordpool.txt', 'r', encoding='utf-8') as f:
            all_words = [line.strip().lower() for line in f.readlines()]
        return np.random.choice(all_words, list_length*2, replace=False)


def generate_prompt(all_stimuli, query_index):
    keys = all_stimuli[:len(all_stimuli)//2]
    values = all_stimuli[len(all_stimuli)//2:]
    prompt = "Extract the value corresponding to the specified key in the pairs below.\n\n{"
    for k,v in zip(keys, values):
        prompt += f"\"{k}\": \"{v}\"\n"
    prompt = prompt[:-2] + "}\n\n" # take off last \n
    prompt += f"Key: \"{keys[query_index]}\"\n"
    prompt += "Corresponding value (only print the value, nothing else): "
    return prompt


def make_pipe(model_id):
    pipe = pipeline("text-generation", model=model_id, dtype=torch.bfloat16, device_map="auto")
    return pipe


def query_model(pipe, inp):
    messages = [{"role": "user", "content": inp}]
    all_outputs = pipe(messages, max_new_tokens=256, temperature=0.7, top_p=0.8, top_k=20,min_p=0.1)
    output = all_outputs[0]['generated_text'][-1]['content'].strip().lower()
    return output, all_outputs


def plot_results(position_accuracy, item_accuracy, savepath=None):
    fig, ax = plt.subplots(1,2, figsize=(18,6), sharey=True, dpi=200)
    plt.suptitle('Serial Key-Value Retrieval Task',fontsize=20)
    ax[0].set_title('Serial Position Effects', fontsize=16)
    sns.lineplot(position_accuracy, marker='o', ax=ax[0])
    ax[0].set_xlabel('Position in List', fontsize=12)
    ax[0].set_ylabel('Mean Recall Accuracy', fontsize=12)
    ax[0].set_xticks(np.linspace(0, list_length, 11))
    ax[0].set_ylim(0, 1) # Accuracy is between 0 and 1
    sns.barplot(x=item_accuracy.keys(), y=item_accuracy.values(), ax=ax[1])
    ax[1].set_title('Item Effects', fontsize=16)
    ax[1].set_xlabel('Stimulus', fontsize=12)
    ax[1].set_xticks([])
    # ax[1].set_xticks(np.arange(len(item_accuracy)), labels=list(item_accuracy.keys()),rotation=45)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()


def run_experiment(list_length, n_experiments, stimulus_type, model_id):
    pipe = make_pipe(model_id)
    stimuli = generate_stimuli(list_length, stimulus_type) # set stimuli once to be used repeatedly

    # store accuracy during experiments
    position_accuracy = np.zeros(list_length)
    item_accuracy = defaultdict(int)
    for _ in tqdm(range(n_experiments)): 
        for query_index in range(list_length): # separate query for each key in the list
            # TO-DO: figure out caching to speed up repeated queries
            prompt = generate_prompt(stimuli, query_index)
            output, _ = query_model(pipe, prompt)
            target_stimulus = stimuli[list_length + query_index]
            position_accuracy[query_index] += output == target_stimulus
            item_accuracy[target_stimulus] += output == target_stimulus
    # calculate accuracy across iterations
    position_accuracy /= n_experiments
    for k, v in item_accuracy.items():
        item_accuracy[k] = v / n_experiments
    item_accuracy = dict(item_accuracy)

    plot_results(position_accuracy, item_accuracy, savepath=f'{dir}/_figures/serialposition_{model_id}_stim-{stimulus_type}_listlength-{list_length}_experiments-{n_experiments}.png')
    return position_accuracy, item_accuracy


if __name__ == '__main__':
    list_length = 200
    stimulus_type = 'uuids'
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    n_experiments = 100
    run_experiment(list_length,  n_experiments, stimulus_type, model_id)