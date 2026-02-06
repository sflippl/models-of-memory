"""
***Memory Inference Experiments***

Train on associative inference task, assess integrative encoding (proactive) vs. retrieval-time inference (reactive)

Paired associate inference setup: goal is to measure A-C association
- Simplest version: AB BC -> A? -> measure output probability of B and C
- Fan in & fan out versions (multiple As to one B, multiple Bs to one A)

Other task ideas (not implemented here):
- Acquired equivalence: AB CB AD -> C?
- Other task: AB CB XY -> A? AB AC XY -> B?

"""

import sys, random, uuid, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
nltk.download('wordnet')
nltk.download('names')
nltk.download('gazetteers')
from nltk.corpus import wordnet, names, gazetteers

# for running on colab
if 'google.colab' in sys.modules:
    os.system("git clone https://github.com/sflippl/models-of-memory.git")
    sys.path.append('models-of-memory')
    dir = 'models-of-memory'
else:
    print("Running locally")
    dir = '.'


def generate_stimuli(all_tokens, n_sets, n_distractor_pairs, fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree, 
                     stimulus_type='words', stimulus_dict=None):
    """
    Each "set" is A->B->C, but strictly its the number of C items (because of fan in/fan out)
    Distractor XY pairs are randomly intermixed if requested
    Fan structure:
        - fan_in_pct: proportion of sets with fan-in structure (multiple As -> one B -> one C)
        - fan_out_pct: proportion of sets with fan-out structure (one A -> multiple Bs -> one C)
        - fan_in_degree: how many A tokens lead to the same B (e.g., 3 means A1->B, A2->B, A3->B->C)
        - fan_out_degree: how many B tokens each A leads to (e.g., 5 means A->B1, A->B2, ..., A->B5, then A->C)
    stimulus_type: 'words', 'names', or 'fakenames'
    stimulus_dict: optional dict with 'A' list and 'BC_pairs' list of tuples for cities/countries mode
    Returns: 
        - list of tuples of training pairs
        - list of possible probes (A)
        - list of lists of direct targets B (may be more than one for fan out)
        - list of indirect targets C (only one per A)
        - list of fan types (simple, fan_in, fan_out)
    """
    if fan_in_pct + fan_out_pct > 1:
        raise ValueError("fan_in_pct + fan_out_pct cannot exceed 1.0")
    
    # determine how many sets of each type
    n_sets_fan_in = int(n_sets * fan_in_pct)
    n_sets_fan_out = int(n_sets * fan_out_pct)
    n_sets_simple = n_sets - n_sets_fan_in - n_sets_fan_out

    # calculate and generate tokens
    n_a_tokens = n_sets_simple + (n_sets_fan_in * fan_in_degree) + n_sets_fan_out
    n_b_tokens = n_sets_simple + n_sets_fan_in + (n_sets_fan_out * fan_out_degree)
    n_c_tokens = n_sets  # one C per set
    n_x_tokens = n_y_tokens = n_distractor_pairs

    if stimulus_dict: 
        a_tokens = np.random.choice(stimulus_dict['A'], n_a_tokens, replace=False)
        # Select matching BC pairs
        bc_indices = np.random.choice(len(stimulus_dict['BC_pairs']), n_sets, replace=False)
        selected_bc = [stimulus_dict['BC_pairs'][i] for i in bc_indices]
        b_tokens = [bc[0] for bc in selected_bc]
        c_tokens = [bc[1] for bc in selected_bc]
        x_tokens = np.random.choice(all_tokens, n_x_tokens, replace=False) 
        y_tokens = np.random.choice(all_tokens, n_y_tokens, replace=False)
    else:
        n_total_tokens = int(n_a_tokens + n_b_tokens + n_c_tokens + n_x_tokens + n_y_tokens)
        tokens = np.random.choice(all_tokens, n_total_tokens, replace=False)
        split_indices = np.cumsum([n_a_tokens, n_b_tokens, n_c_tokens, n_x_tokens, n_y_tokens], dtype=int)[:-1]
        a_tokens, b_tokens, c_tokens, x_tokens, y_tokens = np.split(tokens, split_indices)

    # build pairs and track mappings
    ab_pairs, bc_pairs, xy_pairs = [],[],[]
    direct_targets = []  # list of lists: each entry corresponds to each A, containing a list of its Bs 
    indirect_targets = []  # list: each entry corresponds to each A, containing its C
    pair_types = [] # track whether each pair is AB, BC, or XY
    fan_types = []  # track whether each A is simple, fan_in, or fan_out
    a_idx, b_idx, c_idx = 0, 0, 0 # keep track of where we are in the tokens

    # simple sets: 1 A -> 1 B -> 1 C
    for _ in range(n_sets_simple):
        ab_pairs.append((a_tokens[a_idx], b_tokens[b_idx]))
        bc_pairs.append((b_tokens[b_idx], c_tokens[c_idx]))
        direct_targets.append([b_tokens[b_idx]])
        indirect_targets.append(c_tokens[c_idx])
        fan_types.append('simple')
        a_idx += 1
        b_idx += 1
        c_idx += 1
    
    # Fan-in sets: multiple As -> 1 B -> 1 C
    for _ in range(n_sets_fan_in): 
        for _ in range(fan_in_degree): # A1->B, A2->B, ...
            ab_pairs.append((a_tokens[a_idx], b_tokens[b_idx]))
            direct_targets.append([b_tokens[b_idx]])
            indirect_targets.append(c_tokens[c_idx])
            fan_types.append('fan_in')
            a_idx += 1
        bc_pairs.append((b_tokens[b_idx], c_tokens[c_idx])) # only one BC pair per set
        b_idx += 1
        c_idx += 1
    
    # Fan-out sets: 1 A -> multiple Bs; only first B -> C
    for _ in range(n_sets_fan_out):
        b_list = []
        for i in range(fan_out_degree): # A->B1, A->B2, ...
            ab_pairs.append((a_tokens[a_idx], b_tokens[b_idx]))
            b_list.append(b_tokens[b_idx])
            if i == 0:  # Only pair the first B with C
                bc_pairs.append((b_tokens[b_idx], c_tokens[c_idx]))
            b_idx += 1
        direct_targets.append(b_list)
        indirect_targets.append(c_tokens[c_idx])
        fan_types.append('fan_out')
        a_idx += 1
        c_idx += 1

    xy_pairs = list(zip(x_tokens, y_tokens))
    
    # Shuffle BC pairs
    bc_perm = np.random.permutation(len(bc_pairs))
    bc_shuffled = [bc_pairs[i] for i in bc_perm]
    
    # Combine training pairs and track types
    train_pairs = ab_pairs + bc_shuffled
    pair_types = ['AB'] * len(ab_pairs) + ['BC'] * len(bc_pairs)
    
    # Insert XY pairs at random locations
    for xy_pair in xy_pairs:
        idx = np.random.randint(0, len(train_pairs) + 1)
        train_pairs.insert(idx, xy_pair)
        pair_types.insert(idx, 'XY')

    return train_pairs, pair_types, fan_types, a_tokens, direct_targets, indirect_targets
    

def generate_prompt(train_pairs, pair_types, test_probe, correct_target=None, foil=None, stimulus_type='words', prompt_type='standard'):
    prompt = ""
    # training pairs
    if stimulus_type == 'words':
        for p in train_pairs:
            prompt += f"{p[0]}->{p[1]} "
    else:
        for p, p_type in zip(train_pairs, pair_types):
            if p_type == 'AB' or ():
                prompt += f"{p[0]} is from {p[1]}. "
            elif p_type == 'BC':
                prompt += f"{p[0]} is in {p[1]}. "
            elif random.random() < 0.5: # XY
                prompt += f"{p[0]} is from {p[1]}. "
            else:
                prompt += f"{p[0]} is in {p[1]}. "

    # Query structure
    if prompt_type == 'standard':
        if stimulus_type == 'words':
            prompt += f"{test_probe}->"
        else:
            prompt += f"{test_probe} is from "
    elif prompt_type == 'AFC':
        choices = [correct_target, foil]
        random.shuffle(choices)
        if stimulus_type == 'words':
            prompt += f"{test_probe}->{choices[0]} or {choices[1]}? "
        else:
            prompt += f"Is {test_probe} from {choices[0]} or {choices[1]}? "
    return prompt


def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def get_single_tokens(tokenizer, arr):
  return [a for a in arr if len( tokenizer(a, add_special_tokens=False)["input_ids"] ) == 1 ]

def get_single_token_nouns(tokenizer):
    """get list of nouns from wordnet that are single tokens in the given model"""
    nouns = [lemma.name() for syn in wordnet.all_synsets("n") for lemma in syn.lemmas()]
    nouns = [n for n in nouns if n.isalpha()]
    return get_single_tokens(tokenizer, nouns)

def get_single_token_names(tokenizer):
    """get list of names from nltk names corpus that are single tokens"""
    all_names = names.words('male.txt') + names.words('female.txt')
    return list(set(get_single_tokens(tokenizer, all_names)))

def get_single_token_geography(tokenizer):
    pairs = [
        ("Paris", "France"), ("Berlin", "Germany"), ("Rome", "Italy"), ("Madrid", "Spain"), ("Lisbon", "Portugal"),
        ("Vienna", "Austria"), ("Brussels", "Belgium"), ("Athens", "Greece"), ("Warsaw", "Poland"), ("Prague", "Czechia"),
        ("Tokyo", "Japan"), ("Seoul", "Korea"), ("Beijing", "China"), ("Bangkok", "Thailand"), ("Hanoi", "Vietnam"),
        ("Delhi", "India"), ("Manila", "Philippines"), ("Jakarta", "Indonesia"), ("Cairo", "Egypt"), ("Nairobi", "Kenya"),
        ("Lagos", "Nigeria"), ("Accra", "Ghana"), ("Tunis", "Tunisia"), ("Algiers", "Algeria"), ("Moscow", "Russia"),
        ("Kyiv", "Ukraine"), ("Oslo", "Norway"), ("Stockholm", "Sweden"), ("Helsinki", "Finland"), ("Copenhagen", "Denmark"),
        ("Dublin", "Ireland"), ("London", "UK"), ("Ottawa", "Canada"), ("Mexico", "Mexico"), ("Havana", "Cuba"),
        ("Kingston", "Jamaica"), ("Panama", "Panama"), ("Bogota", "Colombia"), ("Quito", "Ecuador"), ("Lima", "Peru"),
        ("Santiago", "Chile"), ("Brasilia", "Brazil"), ("Caracas", "Venezuela"), ("Sydney", "Australia"), ("Suva", "Fiji"),
        ("Amman", "Jordan"), ("Beirut", "Lebanon"), ("Baghdad", "Iraq"), ("Tehran", "Iran"), ("Riyadh", "Saudi"),
        ("Kuwait", "Kuwait"), ("Doha", "Qatar"), ("Muscat", "Oman"), ("Kabul", "Afghanistan"), ("Islamabad", "Pakistan")
    ]
    single_token_pairs = []
    for city, country in pairs:
        country_ids = tokenizer(country, add_special_tokens=False)["input_ids"] # this is the only one that needs to be single-token
        if len(country_ids) == 1:
            single_token_pairs.append((city, country))
    return single_token_pairs


def get_fictional_single_tokens(tokenizer, count):
    """
    Scavenges the model's own vocabulary for strings it treats as atomic 
    but that are not recognized as real English nouns.
    """
    # Get all potential strings from the model's vocabulary
    # (Handling various tokenizer formats like GPT, Llama, and BERT)
    vocab = tokenizer.get_vocab().keys()
    fictional_pool = []
    
    for t in vocab:
        # 1. Clean up subword/whitespace markers (like 'Ġ', ' ', '##')
        clean = t.replace('Ġ', '').replace(' ', '').replace('##', '')
        
        # 2. Heuristics for a "Country" name (alphabetic, 5-8 chars)
        if clean.isalpha() and 5 <= len(clean) <= 8:
            # 3. Validation: Verify it stays a single token and isn't a real word
            if len(tokenizer.encode(clean, add_special_tokens=False)) == 1:
                if not wordnet.synsets(clean.lower()):
                    fictional_pool.append(clean.capitalize())
        
        if len(fictional_pool) >= count + 50: # Get a buffer then stop
            break
            
    return random.sample(fictional_pool, min(count, len(fictional_pool)))


def query_model(model, tokenizer, prompt, target_tokens=None):
    """
    Get logits, probabilities, and ranks for specific target tokens.
    If target_tokens is None, returns top 10 ranked tokens.
    Returns dict mapping each target token to {logit, prob, rank}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]  # logits for next token
    probs = torch.softmax(next_token_logits, dim=-1)  # probabilites for next token
    
    if target_tokens is None:
        top_probs, top_indices = torch.topk(probs, 10)
        results = {}
        for i in range(10):
            token_id = top_indices[i].item()
            token = tokenizer.decode([token_id])
            results[token] = {
                'logit': next_token_logits[token_id].item(), 
                'prob': top_probs[i].item(), 
                'rank': i + 1
            }
        return results

    target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    results = {}
    for token, token_id in zip(target_tokens, target_token_ids):
        results[token] = {
            'logit': next_token_logits[token_id].item(), 
            'prob': probs[token_id].item(), 
            'rank': (probs > probs[token_id]).sum().item() + 1
        }
    return results



def plot_results(results_df, savepath=None):
    
    fan_types = results_df['fan_type'].unique()
    n_rows = len(fan_types)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows), dpi=200)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    color_b = 'royalblue'  # blue for direct
    color_c = 'forestgreen'  # green for indirect
    
    for row, fan_type in enumerate(fan_types):
        # Direct targets (B) - left column
        ax_b = axes[row, 0]
        ax_b_rank = ax_b.twinx()
        
        b_data = results_df[(results_df['fan_type'] == fan_type) & (results_df['target_type'] == 'direct')]
        probs_b = b_data['probability'].values
        ranks_b = b_data['rank'].values
        
        # Violin plots
        parts_prob = ax_b.violinplot([probs_b], positions=[0.3], widths=0.25, showmeans=True)
        for pc in parts_prob['bodies']:
            pc.set_facecolor(color_b)
            pc.set_alpha(0.5)
        parts_rank = ax_b_rank.violinplot([ranks_b], positions=[0.7], widths=0.25, showmeans=True)
        for pc in parts_rank['bodies']:
            pc.set_facecolor(color_b)
            pc.set_alpha(0.5)
        
        # Scatter points
        ax_b.scatter(np.zeros(len(probs_b)) + 0.3 + np.random.normal(0, 0.02, len(probs_b)), 
                     probs_b, alpha=0.4, s=15, color=color_b)
        ax_b_rank.scatter(np.zeros(len(ranks_b)) + 0.7 + np.random.normal(0, 0.02, len(ranks_b)), 
                          ranks_b, alpha=0.4, s=15, color=color_b)
        
        ax_b.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax_b_rank.set_ylabel('Rank', fontsize=11, fontweight='bold')
        ax_b.set_xticks([])
        ax_b.set_title(f'{fan_type.replace("_", " ").title()} - Direct Targets (B)', fontsize=12, fontweight='bold')
        ax_b.grid(axis='y', alpha=0.2)
        
        # Indirect targets (C) - right column
        ax_c = axes[row, 1]
        ax_c_rank = ax_c.twinx()
        
        c_data = results_df[(results_df['fan_type'] == fan_type) & (results_df['target_type'] == 'indirect')]
        probs_c = c_data['probability'].values
        ranks_c = c_data['rank'].values
        
        # Violin plots
        parts_prob = ax_c.violinplot([probs_c], positions=[0.3], widths=0.25, showmeans=True)
        for pc in parts_prob['bodies']:
            pc.set_facecolor(color_c)
            pc.set_alpha(0.5)
        parts_rank = ax_c_rank.violinplot([ranks_c], positions=[0.7], widths=0.25, showmeans=True)
        for pc in parts_rank['bodies']:
            pc.set_facecolor(color_c)
            pc.set_alpha(0.5)
        
        # Scatter points
        ax_c.scatter(np.zeros(len(probs_c)) + 0.3 + np.random.normal(0, 0.02, len(probs_c)), 
                     probs_c, alpha=0.4, s=15, color=color_c)
        ax_c_rank.scatter(np.zeros(len(ranks_c)) + 0.7 + np.random.normal(0, 0.02, len(ranks_c)), 
                          ranks_c, alpha=0.4, s=15, color=color_c)
        
        ax_c.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax_c_rank.set_ylabel('Rank', fontsize=11, fontweight='bold')
        ax_c.set_xticks([])
        ax_c.set_title(f'{fan_type.replace("_", " ").title()} - Indirect Targets (C)', fontsize=12, fontweight='bold')
        ax_c.grid(axis='y', alpha=0.2)
    
    plt.suptitle('Associative Inference: Next Token Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def run_experiment(n_sets, model_id, n_distractor_pairs, fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree, stimulus_type='words', prompt_type='standard'):    
    model, tokenizer = load_model(model_id)
    
    if stimulus_type == 'words':
        all_tokens = get_single_token_nouns(tokenizer)
    elif stimulus_type == 'names':
        names = get_single_token_names(tokenizer)
        geo_pairs = get_single_token_geography(tokenizer)
        
    elif stimulus_type == 'geography':
        all_tokens = get_single_token_geography(tokenizer)
    
    stimulus_dict = None
    if 'names' in stimulus_type:
        names_list = get_single_token_names(tokenizer)
        if 'fake' in stimulus_type:
            # For fictional, we still generate them separately but we can pair them
            cities_list = get_fictional_tokens(tokenizer, 50) 
            countries_list = get_fictional_tokens(tokenizer, 50)
            bc_pairs = list(zip(cities_list, countries_list))
        else:
            bc_pairs = get_single_token_geographic_pairs(tokenizer)
        
        stimulus_dict = {
            'A': names_list,
            'BC_pairs': bc_pairs
        }

    training_data = generate_stimuli(
        all_tokens, n_sets, n_distractor_pairs, fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree,
        stimulus_type=stimulus_type, stimulus_dict=stimulus_dict
    )
    train_pairs, pair_types, fan_types, test_probes, test_direct_targets, test_indirect_targets = training_data

    results_list = []
    for i, (probe, direct_targets, indirect_target, fan_type) in enumerate(zip(test_probes, test_direct_targets, test_indirect_targets, fan_types)):
        # Pick foil for AFC
        foil = None
        if prompt_type == 'AFC':
            # Pick a target from a DIFFERENT set
            other_targets = [t for j, t in enumerate(test_indirect_targets) if j != i and t != indirect_target]
            foil = random.choice(other_targets) if other_targets else "Unknown"

        prompt = generate_prompt(train_pairs, pair_types, probe, correct_target=indirect_target, foil=foil, 
                                 stimulus_type=stimulus_type, prompt_type=prompt_type)
        
        # Targets for probability check
        all_targets = direct_targets + [indirect_target]
        if foil and foil not in all_targets:
            all_targets.append(foil)
            
        results = query_model(model, tokenizer, prompt, all_targets)
        
        # Log B results
        for b in direct_targets:
            results_list.append({
                'fan_type': fan_type, 'target_type': 'direct', 'rank': results[b]['rank'],
                'logit': results[b]['logit'], 'probability': results[b]['prob']
            })
        
        # Log C results
        results_list.append({
            'fan_type': fan_type, 'target_type': 'indirect', 'rank': results[indirect_target]['rank'],
            'logit': results[indirect_target]['logit'], 'probability': results[indirect_target]['prob'],
            'foil_prob': results[foil]['prob'] if foil else None
        })
    
    results = pd.DataFrame(results_list)
    plot_results(results, savepath=f'{dir}/_figures/inference_{model_id.replace("/", "-")}_nsets-{n_sets}_ndistractorpairs-{n_distractor_pairs}_faninpct-{fan_in_pct}_fanoutpct-{fan_out_pct}_fanindegree-{fan_in_degree}_fanoutdegree-{fan_out_degree}_stim-{stimulus_type}_prompt-{prompt_type}.png')
    return results


if __name__ == '__main__':
    n_sets = 10 # how many C items (BC pairs)
    n_distractor_pairs = 0 # how many XY pairs
    fan_in_pct = 0.0 # percentage of sets that fan in
    fan_out_pct = 0.0 # percentage of sets that fan out
    fan_in_degree = 2 # how many As per B
    fan_out_degree = 2 # how many Bs per A
    model_id = "Qwen/Qwen3-4B"
    run_experiment(
        n_sets, model_id, n_distractor_pairs, 
        fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree, 
        stimulus_type='abstract'
    )