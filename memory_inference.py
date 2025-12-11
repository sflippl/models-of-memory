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
from nltk.corpus import wordnet as wn

# for running on colab
if 'google.colab' in sys.modules:
    os.system("git clone https://github.com/sflippl/models-of-memory.git")
    sys.path.append('models-of-memory')
    dir = 'models-of-memory'
else:
    print("Running locally")
    dir = '.'


def generate_stimuli(all_tokens, n_sets, n_distractor_pairs, fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree):
    """
    Each "set" is A->B->C, but strictly its the number of C items (because of fan in/fan out)
    Distractor XY pairs are randomly intermixed if requested
    Fan structure:
        - fan_in_pct: proportion of sets with fan-in structure (multiple As -> one B -> one C)
        - fan_out_pct: proportion of sets with fan-out structure (one A -> multiple Bs -> one C)
        - fan_in_degree: how many A tokens lead to the same B (e.g., 3 means A1->B, A2->B, A3->B->C)
        - fan_out_degree: how many B tokens each A leads to (e.g., 5 means A->B1, A->B2, ..., A->B5, then A->C)
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
    n_total_tokens = int(n_a_tokens + n_b_tokens + n_c_tokens + n_x_tokens + n_y_tokens)

    tokens = np.random.choice(all_tokens, n_total_tokens, replace=False)
    split_indices = np.cumsum([n_a_tokens, n_b_tokens, n_c_tokens, n_x_tokens, n_y_tokens], dtype=int)[:-1]
    a_tokens, b_tokens, c_tokens, x_tokens, y_tokens = np.split(tokens, split_indices)

    # build pairs and track mappings
    ab_pairs, bc_pairs, xy_pairs = [],[],[]
    direct_targets = []  # list of lists: each entry corresponds to each A, containing a list of its Bs 
    indirect_targets = []  # list: each entry corresponds to each A, containing its C
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
    # Combine all training pairs
    train_pairs = ab_pairs + list(np.random.permutation(bc_pairs)) # shuffle BC pairs so they don't appear in the same order as AB pairs (but always come after)
    for xy_pair in xy_pairs: # insert XY pairs at random locations
        train_pairs.insert(np.random.randint(0, len(train_pairs) + 1), xy_pair)

    return train_pairs, a_tokens, direct_targets, indirect_targets, fan_types
    

def generate_prompt(train_pairs, test_probe):
    prompt = ""
    for token1, token2 in train_pairs:
        prompt += f"{token1}->{token2} "
    prompt += f"{test_probe}->"
    return prompt


def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def get_single_token_nouns(tokenizer):
    """get list of nouns from wordnet that are single tokens in the given model"""
    nouns = [lemma.name() for syn in wn.all_synsets("n") for lemma in syn.lemmas()]
    nouns = [n for n in nouns if n.isalpha()]
    
    single_token_nouns = []
    for w in nouns:
        # use add_special_tokens=False to avoid BOS/EOS etc.
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            single_token_nouns.append(w)
    
    return single_token_nouns



def query_model(model, tokenizer, prompt, target_tokens):
    """
    Get logits, probabilities, and ranks for specific target tokens.
    Returns dict mapping each target token to {logit, prob, rank}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]  # logits for next token
    probs = torch.softmax(next_token_logits, dim=-1)  # probabilites for next token
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


def run_experiment(n_sets, model_id, n_distractor_pairs, fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree):    
    model, tokenizer = load_model(model_id)
    all_tokens = get_single_token_nouns(tokenizer)

    training_pairs, test_probes, test_direct_targets, test_indirect_targets, fan_types = generate_stimuli(
        all_tokens,
        n_sets,
        n_distractor_pairs,
        fan_in_pct,
        fan_out_pct,
        fan_in_degree,
        fan_out_degree
    )

    # Build DataFrame to store all results
    results_list = []

    for probe, direct_targets, indirect_target, fan_type in zip(test_probes, test_direct_targets, test_indirect_targets, fan_types):
        prompt = generate_prompt(training_pairs, probe)
        
        # Query model, get logits/probs of all B and C items at once
        all_targets = direct_targets + [indirect_target]
        results = query_model(model, tokenizer, prompt, all_targets)
        
        # Add B (direct) results
        for b in direct_targets:
            results_list.append({
                'fan_type': fan_type,
                'target_type': 'direct',
                'rank': results[b]['rank'],
                'logit': results[b]['logit'],
                'probability': results[b]['prob']
            })
        
        # Add C (indirect) result
        results_list.append({
            'fan_type': fan_type,
            'target_type': 'indirect',
            'rank': results[indirect_target]['rank'],
            'logit': results[indirect_target]['logit'],
            'probability': results[indirect_target]['prob']
        })
    
    results = pd.DataFrame(results_list)
    plot_results(results, savepath=f'{dir}/_figures/inference_{model_id.replace("/", "-")}_nsets-{n_sets}_ndistractorpairs-{n_distractor_pairs}_faninpct-{fan_in_pct}_fanoutpct-{fan_out_pct}_fanindegree-{fan_in_degree}_fanoutdegree-{fan_out_degree}.png')
    return results


if __name__ == '__main__':
    n_sets = 10 # how many C items (BC pairs)
    n_distractor_pairs = 0 # how many XY pairs
    fan_in_pct = 0.0 # percentage of sets that fan in
    fan_out_pct = 0.0 # percentage of sets that fan out
    fan_in_degree = 2 # how many As per B
    fan_out_degree = 2 # how many Bs per A
    model_id = "Qwen/Qwen3-4B"
    run_experiment(n_sets, model_id, n_distractor_pairs, fan_in_pct, fan_out_pct, fan_in_degree, fan_out_degree)