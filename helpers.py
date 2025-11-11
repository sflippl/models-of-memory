import random
import numpy as np
import torch
from transformers import pipeline


def txt_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().lower() for line in f.readlines()]
    return lines


def subsample_words(words, n, seed=None, avoid=[]):
    """
    Return a random subset of n unique words from the list.
    Optionally set a random seed for reproducibility.
    """
    if n > len(words):
        raise ValueError("Sample size cannot exceed number of words.")
    if seed is not None:
        random.seed(seed)
    if len(avoid) > 0:
        words = [w for w in words if w not in avoid]
    return random.sample(words, n)


def make_pairs(first_words, second_words, joiner='-', randomize=False):
    # "chair-apple"
    np.random.seed(0)
    rand_draws = np.random.rand(len(first_words))
    pairs = []
    for first, second, rand in zip(first_words, second_words, rand_draws):
        if randomize and rand < 0.5:
            first, second = second, first
        pairs.append(f'{first}{joiner}{second}')
    return pairs


def make_test_probes(cue_words, target_words, lure_words):
    # for paired associate tests: 2-alternative-forced-choice tests
    np.random.seed(0)
    perm = np.random.permutation(len(cue_words))
    cue_words = np.array(cue_words)[perm]
    target_words = np.array(target_words)[perm]
    lure_words = np.array(lure_words)[perm]
    out = []
    rand_draw = np.random.rand(len(cue_words))
    for i,(c,t,l) in enumerate(zip(cue_words, target_words, lure_words)):
        if rand_draw[i] < 0.5:
            out.append(f'{c}: {t} or {l}?')
        else:
            out.append(f'{c}: {l} or {t}?')
    return out, list(target_words), list(lure_words)


def make_training_order(first_pairs, second_pairs, unrelated_pairs):
    # for paired associate inference nad acquired equivalence
    # AB[i] must come before BC[i], but DE can come anywhere
    all_items = first_pairs + second_pairs + unrelated_pairs
    random.seed(0)
    random.shuffle(all_items)
    # now loop through random list and replace any BC that are before their AB pair
    for i, (first, second) in enumerate(zip(first_pairs, second_pairs)):
        first_idx = all_items.index(first)
        second_idx = all_items.index(second)
        if second_idx < first_idx:
            # remove second and reinsert it after first at a random later position
            all_items.pop(second_idx)
            first_idx = all_items.index(first)
            insert_pos = random.randint(first_idx + 1, len(all_items))
            all_items.insert(insert_pos, second)
    return all_items


def make_inp(words, test_words=[], preamble='', cue='', interjection=None, interjection_words=[]):
    inp = f'{preamble} {", ".join(words)}. '
    if interjection:
        inp += f'{interjection} {", ".join(interjection_words)}. '
    inp += cue
    if len(test_words) > 0:
        inp += ", ".join(test_words)
    return inp


def make_pipe(model_id):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe


def query_model_helper(pipe, inp):
    messages = [
    {"role": "user", "content": inp},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.

    )
    return outputs


def query_model(pipe, inp):
    outputs = query_model_helper(pipe, inp)
    lst_words = outputs[0]["generated_text"][-1]['content'].split('— Wait')[0].split(', ')
    lst_words = [w.strip().lower() for w in lst_words]
    return lst_words