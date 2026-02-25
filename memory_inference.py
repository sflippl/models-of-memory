"""
MEMORY INFERENCE EXPERIMENTS
This implements something like a paired-associate inference experiment.
Given A-R1-B and B-R2-C we test if the model can infer A-R1+R2-C.

USAGE EXAMPLE:
the main function is generate_trials(), which works as follows:
Arguments:
    - stim_type
        OPTIONS: 'single_token_nouns', 'real_geography', 'wrong_geography', 'fake_geography', 'scifi_geography', 'properties', 'formal'
        See below for explanation.
    - n_trials (each trial is a unique prompt)
    - tokenizer (only needed for single-token noun stimuli)
    - fan_type ('in' or 'out')
    - fan_degree (number of A's per B for 'in', number of B's per A for 'out')
    - n_sets (number of A-B-C sets)
    - add_ac_pairs (whether to add A-C pairs from other sets into the prompt)
Returns:
    list of dict, each dict is a "trial" with keys 'prompt', 
    'target_c', 'distractor_c', 'target_b', 'distractor_b', 
    'target_a', 'a_items', 'b_items', 'c_items'

You can also use testing_helper(model, tokenizer) to run a bunch of trials and compute accuracy.
But generate_trials() will work to make your own manual loop.

STIMULUS TYPES:
    - "single_token_nouns": chair->trumpet trumpet->cheese chair->
    - "real_geography": Sam is from Paris. Paris is in France. Sam is from the country of
    - "wrong_geography": Sam is from Paris. Paris is in Germany. Sam is from the country of
    - "fake_geography": Sam is from Fubalu. Fubalu is in France. Sam is from the country of
    - "scifi_geography": Sam is from Arakeen. Arakeen is in France. Sam is from the country of
    - "properties": Fubalu is made from glarp. Glarp has the property of translucense. Fubalu has the property of 
    - "formal": A1 R1 B1  B1 R2 C1  A1 R1+R2 C1

If you'd like to manually edit the stimuli or relations, you can do so in the "stimulus generator" functions.
"""

import random
import string
import numpy as np
import nltk
from nltk.corpus import wordnet
from typing import List, Tuple, Dict, Any, Optional
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
def setup_nltk():
    try:
        wordnet.all_synsets('n')
    except LookupError:
        nltk.download('wordnet')
setup_nltk()
from stimuli import NAMES, CITY_COUNTRY_PAIRS, SCIFI_LOCATIONS, PROPERTIES


# --- Stimulus Generator Helpers (Pools) ---

def get_fake_words(n, length=5, seed=None):
    """Pronounceable fake words alternating consonants and vowels"""
    if seed is not None:
        random.seed(seed)
    vowels = "aeiou"
    consonants = "".join(set(string.ascii_lowercase) - set(vowels))
    words = []
    while len(words) < n:
        word = ""
        for i in range(length):
            if i % 2 == 0:
                word += random.choice(consonants)
            else:
                word += random.choice(vowels)
        if word not in words:
            words.append(word.capitalize())
    return words

def get_single_token_nouns(tokenizer, n):
    """Single token nouns from wordnet in a given tokenizer"""
    nouns = list(set([lemma.name() for syn in wordnet.all_synsets("n") for lemma in syn.lemmas() if lemma.name().isalpha()]))
    if tokenizer:
        single_tokens = [n for n in nouns if len(tokenizer.encode(n, add_special_tokens=False)) == 1]
    else:
        single_tokens = [n for n in nouns if len(n) < 6]
    return random.sample(single_tokens, n)

def get_names(n):
    """Names from stimuli module"""
    return random.sample(NAMES, n)

def get_scifi_cities(n):
    """Sci-Fi locations from stimuli module"""
    return random.sample(SCIFI_LOCATIONS, n)

def get_cities_countries(n, swap_pairs=False):
    """Verified City-Country pairs from stimuli module"""
    selected_pairs = random.sample(CITY_COUNTRY_PAIRS, n)
    cities = [p[0] for p in selected_pairs]
    countries = [p[1] for p in selected_pairs]
    if swap_pairs:
        # Guarantee no city matches its country
        shuffled_countries = countries[:]
        if len(countries) > 1:
            while any(shuffled_countries[i] == countries[i] for i in range(len(countries))):
                random.shuffle(shuffled_countries)
        countries = shuffled_countries
    return cities, countries


# --- Stimulus Generators (return A, B, C, relations) ---

def get_noun_stimuli(tokenizer, n):
    # Need 3n unique nouns total for A, B, C
    pool = get_single_token_nouns(tokenizer, 3*n)
    return pool[:n], pool[n:2*n], pool[2*n:3*n], [" -> ", " -> ", " -> "]

def get_real_geography_stimuli(n):
    a = get_names(n)
    b, c = get_cities_countries(n, swap_pairs=False)
    return a, b, c, [" is from ", " is in ", " is from the country of "]

def get_wrong_geography_stimuli(n):
    a = get_names(n)
    b, c = get_cities_countries(n, swap_pairs=True)
    return a, b, c, [" is from ", " is in ", " is from the country of "]

def get_fake_geography_stimuli(n):
    a = get_names(n)
    b = get_fake_words(n, length=6)
    _, c = get_cities_countries(n)
    return a, b, c, [" is from ", " is in ", " is from the country of "]

def get_scifi_geography_stimuli(n):
    a = get_names(n)
    b = get_scifi_cities(n)
    _, c = get_cities_countries(n) # Random real countries
    return a, b, c, [" is from ", " is in ", " is from the country of "]

def get_object_property_stimuli(n):
    # Need 2n fake words for objects and materials
    words = get_fake_words(2*n, length=6)
    objs = words[:n]
    mats = words[n:]
    prop_list = random.sample(PROPERTIES, n)
    return objs, mats, prop_list, [" is made of ", " has the property of ", " has the property of "]

def get_formal_stimuli(n, offset=0):
    a_items = [f"A{i+offset}" for i in range(1, n + 1)]
    b_items = [f"B{i+offset}" for i in range(1, n + 1)]
    c_items = [f"C{i+offset}" for i in range(1, n + 1)]
    relations = [" R1 ", " R2 ", " (R1+R2) "]
    return a_items, b_items, c_items, relations


# --- Prompt Assembly Logic ---

def assemble_inference_prompt(a_items, b_items, c_items, relations, 
    fan_type='none', fan_degree=1, add_ac_pairs=False, n_sets=1):

    r1, r2, r1r2 = relations
    ab_pairs = []
    bc_pairs = []
    ac_map = {}
    ab_map = {}
    
    if fan_type == 'in':
        # Many A's to one B. B -> C is 1:1.
        for i in range(n_sets):
            b = b_items[i]
            c = c_items[i]
            bc_pairs.append((b, c))
            for j in range(fan_degree):
                idx = i * fan_degree + j
                if idx < len(a_items):
                    a = a_items[idx]
                    ab_pairs.append((a, b))
                    ac_map[a] = c
                    ab_map[a] = b
                
    elif fan_type == 'out':
        # One A to many B's. Each B -> C is 1:1.
        for i in range(n_sets):
            if i < len(a_items) and i < len(c_items):
                a = a_items[i]
                c = c_items[i]
                current_cities = []
                for j in range(fan_degree):
                    idx = i * fan_degree + j
                    if idx < len(b_items):
                        b = b_items[idx]
                        ab_pairs.append((a, b))
                        bc_pairs.append((b, c))
                        current_cities.append(b)
                ac_map[a] = c
                ab_map[a] = current_cities
            
    else: # none
        for i in range(n_sets):
            if i < len(a_items) and i < len(b_items) and i < len(c_items):
                a, b, c = a_items[i], b_items[i], c_items[i]
                ab_pairs.append((a, b))
                bc_pairs.append((b, c))
                ac_map[a] = c
                ab_map[a] = b

    random.shuffle(ab_pairs)
    random.shuffle(bc_pairs)
    
    # Target Selection (use first a_item as target)
    target_a = a_items[0]
    target_c = ac_map.get(target_a, "")
    target_b = "" if fan_type == 'out' else ab_map.get(target_a, "")
    
    # Distractor Selection (only if n_sets == 2)
    distractor_c = [c for c in c_items if c != target_c][0] if n_sets == 2 else ""
    # Find a city associated with the distractor country
    distractor_b = [b for b, c in bc_pairs if c == distractor_c][0] if n_sets == 2 and distractor_c else ""
    
    sentences = []
    for a, b in ab_pairs:
        sentences.append(f"{a}{r1}{b}.")
    for b, c in bc_pairs:
        sentences.append(f"{b}{r2}{c}.")
        
    if add_ac_pairs:
        for a, c in ac_map.items():
            if a != target_a:
                sentences.append(f"{a}{r1r2}{c}.")
    
    prompt_body = " ".join(sentences)
    prompt = f"{prompt_body} Therefore, {target_a}{r1r2}"
    
    return {
        "prompt": prompt,
        "target": target_c,
        "target_c": target_c,
        "distractor_c": distractor_c,
        "target_b": target_b,
        "distractor_b": distractor_b,
        "target_a": target_a,
        "a_items": a_items,
        "b_items": b_items,
        "c_items": c_items
    }

def get_stimuli_by_type(stim_type, n, tokenizer=None, seed=None, offset=0):
    if seed is not None: random.seed(seed)
    if stim_type == "formal":
        return get_formal_stimuli(n, offset=offset)
    elif stim_type == "single_token_nouns":
        return get_noun_stimuli(tokenizer, n)
    elif stim_type == "real_geography":
        return get_real_geography_stimuli(n)
    elif stim_type == "wrong_geography":
        return get_wrong_geography_stimuli(n)
    elif stim_type == "scifi_geography":
        return get_scifi_geography_stimuli(n)
    elif stim_type == "fake_geography":
        return get_fake_geography_stimuli(n)
    elif stim_type == "properties":
        return get_object_property_stimuli(n)
    else:
        raise ValueError(f"Unknown stim_type: {stim_type}")


def generate_trials(stim_type, n_trials=100, tokenizer=None, 
    fan_type='none', fan_degree=1, n_sets=2, add_ac_pairs=False):

    trials = []
    for i in range(n_trials):
        n_needed = max(n_sets * fan_degree, n_sets)
        a, b, c, rels = get_stimuli_by_type(stim_type, n_needed, tokenizer, seed=i, offset=i*n_needed)
        trial = assemble_inference_prompt(
            a, b, c, rels,
            fan_type=fan_type,
            fan_degree=fan_degree,
            add_ac_pairs=add_ac_pairs,
            n_sets=n_sets
        )
        trials.append(trial)
    return trials


def load_model(model_id):
    """Load model and tokenizer with bfloat16"""
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

# helpers for assessing correctness
def x_before_y(text: str, x: str, y: str) -> bool:
    """Check if string x appears before y in text"""
    x_pos = text.find(x)
    y_pos = text.find(y)
    if x_pos == -1: return False
    if y_pos == -1: return True
    return x_pos < y_pos

def correct_helper(df):
    """Decide if output is correct: is target token first, or does it come before distractor/direct"""
    target_first = np.array([
        outp.split()[0].translate(str.maketrans('', '', string.punctuation)) == target_c 
        for outp, target_c in zip(df['outp'], df['target_c'])
    ])
    target_before_distractor = np.array([
        x_before_y(outp, target_c, distractor_c) 
        for outp, target_c, distractor_c in zip(df['outp'], df['target_c'], df['distractor_c'])
    ])
    target_before_direct = np.array([
        x_before_y(outp, target_c, target_b) 
        for outp, target_c, target_b in zip(df['outp'], df['target_c'], df['target_b'])
    ])
    df['correct'] = target_first | target_before_distractor
    df['direct'] = target_first | target_before_direct
    return df


### run some trials
def testing_helper(model, tokenizer, stim_type="scifi", n_trials=100, **kwargs):
    """Run model inference on a set of trials and return evaluation results"""
    dct = {
        'outp': [],
        'target_c': [],
        'distractor_c': [],
        'target_b': [],
        'distractor_b': []
    }
    for _ in tqdm(range(n_trials)):
        trial = generate_trials(stim_type, n_trials=1, tokenizer=tokenizer, **kwargs)[0]
        inputs = tokenizer.encode(trial['prompt'], return_tensors='pt').to(model.device)
        outp_ids = model.generate(inputs, max_new_tokens=10)
        outp = tokenizer.batch_decode(outp_ids[:, inputs.shape[-1]:], skip_special_tokens=True)
        
        dct['outp'].append(outp)
        dct['target_c'].append(trial['target_c'])
        dct['distractor_c'].append(trial['distractor_c'])
        dct['target_b'].append(trial['target_b'])
        dct['distractor_b'].append(trial['distractor_b'])
        
    df = pd.DataFrame(dct)
    df['outp'] = [x[0] for x in df['outp']]
    return correct_helper(df)


if __name__ == "__main__":
    # Example usage:
    # model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
    # results = testing_helper(model, tokenizer, stim_type="scifi", n_sets=2, fan_type='in', fan_degree=2)
    # print(results.correct.mean())
    pass
