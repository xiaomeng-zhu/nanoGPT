import os, csv, argparse, random, json, logging, gc
from collections import defaultdict

import torch
import tiktoken
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import pickle
from pyinflect import getAllInflections

from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import combinations
from datasets import get_dataset_config_names, load_dataset, load_from_disk # huggingface
from math import exp

from model import GPTConfig, GPT

blimp_configs = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

parser = argparse.ArgumentParser()

# Tokenizer Parameters
parser.add_argument("--model_seed", type=int, help="random seed of model", default=42)
parser.add_argument("--model_type", type=str, help="pretrained or self-trained", default="self-trained")
parser.add_argument("--model_size", type=int, help="size of pretrained model in M", default=14)
parser.add_argument("--model_name", type=str, help="name of pretrained model in M", default="pythia")
parser.add_argument("--data_seed", type=int, help="random seed of data", default=42)
parser.add_argument("--subsplit", type=int, help="subsplit of model", default=999)
parser.add_argument("--step", type=int, help="num iterations of the checkpoint", default=0) # not required if evaluating from pretrained
args = parser.parse_args()

log_file_name = f"blimp_eval-m{args.model_seed}-d{args.data_seed}-{args.subsplit}-step={args.step}"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(),logging.FileHandler("logs/" + log_file_name + ".log")])
logging = logging.getLogger(__name__)
logging.info(args)

enc = tiktoken.get_encoding("gpt2")
pythia_tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{args.model_size}m")
if args.model_name != "pythia" and args.model_type == "pretrained": # other pretrained models use the generic tokenizer
    generic_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# if args.model_type == "pretrained":
#     # only do these import if pretrained and non-pythia
#     from transformers import GPTOssForCausalLM, AutoConfig, PreTrainedTokenizerFast, GPT2LMHeadModel, RobertaTokenizerFast, GPTNeoXForCausalLM

def process_buggy(example):
    if args.model_type == "pretrained" and args.model_name == "pythia": # if pretrained and pythia
        # logging.info(f"Using pretrained pythia tokenizer")
        tokenizer_output = pythia_tokenizer(example, 
                               return_tensors="pt").to(DEVICE)
        input_ids = tokenizer_output["input_ids"].to(DEVICE)

    elif args.model_type == "pretrained" and args.model_name != "pythia": # if pretrained but not pythia:
        ids = generic_tokenizer(example, return_tensors="pt").to(DEVICE)
    else:
        ids = enc.encode_ordinary(example) # encode_ordinary ignores any special tokens
        ids = torch.tensor([ids]).to(DEVICE)
    return input_ids

def process(example):
    if args.model_type == "pretrained" and args.model_name == "pythia": # if pretrained and pythia
        # logging.info(f"Using pretrained pythia tokenizer")
        tokenizer_output = pythia_tokenizer(example, 
                               return_tensors="pt").to(DEVICE)
        input_ids = tokenizer_output["input_ids"].to(DEVICE)

        # prepend the bos token id
        bos = pythia_tokenizer.bos_token_id # bos token id is 0 in Pythia as the eos token id
        bos_tensor = torch.tensor([[bos]], device=DEVICE)
        input_with_bos = torch.cat([bos_tensor, input_ids], dim=1).to(DEVICE)

    elif args.model_type == "pretrained" and args.model_name != "pythia": # if pretrained but not pythia:
        ids = generic_tokenizer(example, return_tensors="pt").to(DEVICE)
        bos = generic_tokenizer.bos_token_id # get the bos token
        bos_tensor = torch.tensor([[bos]], device=DEVICE)
        input_with_bos = torch.cat([bos_tensor, input_ids], dim=1).to(DEVICE) # concatenate
        ids = enc.encode_ordinary(example) # encode_ordinary ignores any special tokens
        ids = torch.tensor([ids]).to(DEVICE)

    return input_with_bos

def load_model(model_seed, data_seed, step, subsplit):
    # load model
    checkpoint = torch.load(f"out/m{model_seed}_d{data_seed}_{subsplit}_step{step}.pt", map_location=torch.device(DEVICE))
    logging.info(f"Loading model checkpoint from out/m{model_seed}_d{data_seed}_{subsplit}_step{step}.pt")
    checkpoint_model_args = checkpoint['model_args']

    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

def download_eval_dataset():
    configs = get_dataset_config_names("nyu-mll/blimp")
    # dataset = load_dataset("nyu-mll/blimp", blimp_config)["train"]
    # dataset.save_to_disk("blimp")
    datasets_dict = {}

    for cfg in configs:
        datasets_dict[cfg] = load_dataset("nyu-mll/blimp", cfg)["train"]

    for cfg, ds in datasets_dict.items():
        ds.save_to_disk(f"blimp_datasets/{cfg}")


def evaluate(dataset):
    res = []
    for d in (dataset):
        good_prob = log_prob(d["sentence_good"])
        bad_prob = log_prob(d["sentence_bad"])
        d["good_prob"] = good_prob
        d["bad_prob"] = bad_prob
        d["correct"] = 1 if good_prob > bad_prob else 0
        res.append(d)
    return res
            

def log_prob(sentence):
    input_ids = process(sentence).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids = input_ids)
        # outputs = model(input_ids, input_ids)
        logits = outputs[0]
        # print(f"logits size {logits.size()}")

        # Shift logits and labels to align
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # print(f"shift labels dim {shift_labels.size()}")

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # print(f"log probs size {log_probs.size()}")

        # Gather log-probability of the correct next token
        # This gives shape (batch_size, seq_len - 1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # print(f"target log probs size {target_log_probs}")

        # Sum to get total log-probability
        total_log_prob = target_log_probs.sum().item()

        # # print out per token prob
        # per_token_log_probs = target_log_probs[0]      # shape: (seq_len - 1,)
        # tokens = input_ids[0, 1:]                     # these are the predicted tokens

        # for token_id, logp in zip(tokens, per_token_log_probs):
        #     token_str = pythia_tokenizer.decode(token_id.item())
        #     print(f"{token_str:>10s}  logP = {logp.item():.4f}")


    return total_log_prob

        
def load_pretrained_model_tokenizer(model_seed, model_name, model_size):
    if model_name == "pythia":
        logging.info(f"Loading model EleutherAI/{model_name}-{model_size}m-seed{model_seed}")
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}-{model_size}m-seed{model_seed}",
            revision="step143000").to(DEVICE)
    elif model_name.startswith("openai"):
        model = GPTOssForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype = torch.bfloat16
        ) # TODO: figure out how to load the openai oss model (will this necessarily involve sharding?)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    return model


if __name__ == "__main__":
    path = "blimp_datasets"
    restored_datasets = {
        cfg: load_from_disk(os.path.join(path, cfg))
        for cfg in os.listdir(path)
    }

    # Load model
    if args.model_type == "pretrained":
        model = load_pretrained_model_tokenizer(args.model_seed, args.model_name, args.model_size)
    else: # if not pretrained model
        model = load_model(args.model_seed, args.data_seed, args.step, args.subsplit)

    # get results
    for conf in tqdm(blimp_configs):
        dataset = restored_datasets[conf]
        res = evaluate(dataset)
        # if args.model_type == "pretrained" and args.model_name == "pythia":
        #     res_fn = f"pretrained_results/blimp_{conf}_{args.model_name}_{args.model_size}M_s{args.model_seed}.csv"
        # elif args.model_type == "pretrained" and args.model_name != "pythia": # not pretrained pythia with random seed
        #     res_fn = f"pretrained_results/blimp_{conf}_{args.model_name.replace('/', '_')}.csv"
        # else:
        #     res_fn = f"results_c4_10M/blimp_{conf}_m{args.model_seed}_d{args.data_seed}_{args.subsplit}_{args.step}.csv"
        # logging.info(f"Saving results to {res_fn}")
        # pd.DataFrame(res).to_csv(res_fn, index=False)
    # download_eval_dataset()