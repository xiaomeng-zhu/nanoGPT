import pandas as pd
import numpy as np
import random
from tqdm import tqdm

SEED = 42

random.seed(SEED)

def no_adjacent_config(trials):
    # make sure that adjacent trials do not share the same configuration
    return all(
        trials[i]["config"] != trials[i-1]["config"]
        for i in range(1, len(trials))
    )

def shuffle_until_valid(trials, max_tries=100000):
    # shuffle the trials until the no adjacent configuration condition is met
    # some smaller numbers will result in value errors, so we are using 100K as the max tries
    trials = list(trials)
    for _ in range(max_tries):
        random.shuffle(trials)
        if no_adjacent_config(trials):
            return trials
    raise ValueError("Could not find valid shuffle without adjacent configs.")

def shuffle_sent(sent):
    # keep the first and the last words in the sentence the same and only shuffle the words in between

    words = sent.split()

    if len(words) <= 3: # if there is less than or equal to three words, then we only reverse the last two words
        assert sent[-1] in [".", "?"]
        words = sent[:-1].split()
        candidate = " ".join([words[0]] + words[1:][::-1]) + sent[-1]
        return candidate
        

    words_to_shuffle = words[1:-1] 
    max_tries = 100
    for _ in range(max_tries):

        # shuffle the words in between
        random.shuffle(words_to_shuffle)

        # prepend and append the words
        candidate = " ".join([words[0]] + words_to_shuffle + [words[-1]])
        if candidate != sent: # make sure that the candidate is not the same as the original sentence
            return candidate
    
    return "ERROR: FIX THIS" # if there are no candidate sentences, return ERROR message in the resulting csv
        

# design decisions for the experiment
blimp_configs = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
configs_per_participant = 10
rows_per_config = 10
n_configs = len(blimp_configs)
n_participants = 67
fillers_per_participant = 10 # 10 fillers with 100 target items

assert n_configs  == 67       # making sure we are covering all 67 blimp configurations


# randomly shuffle the blimp configs list because currently neighboring configs are related
configs_order = blimp_configs[:]
random.shuffle(configs_order)
assert configs_order != blimp_configs # make sure that the order of the configuration is actually shuffled

# create participant to configuration assignment
# p1 gets 1 2 3 4 5; p2 gets 2 3 4 5 6 etc. so after a round of 67, configs are counterbalanced
participant_to_configs = []
for p in range(n_participants):
    assigned = [configs_order[(p + t) % n_configs] for t in range(configs_per_participant)]
    participant_to_configs.append(assigned)


keep_cols = ["sentence_good", "sentence_bad", "field", "linguistics_term", "UID", "pair_id"]

# sample 20 rows per configuration
config_sampled = {}
for config in blimp_configs:
    df = pd.read_csv(f"results_babylm/blimp_{config}_m42_d42_0_500.csv", usecols=keep_cols)
    config_sampled[config] = df.sample(n=rows_per_config, random_state=SEED).copy()


# create master spreadsheet
rows = []

# iterate through each participant
for p in tqdm(range(n_participants)):
    # get configurations assigned to participant
    assigned_configs = participant_to_configs[p]
    excluded = set(assigned_configs)
    # configurations that can be used as fillers
    filler_pool_configs = [c for c in blimp_configs if c not in excluded]

    # ========= target =========
    # enumerate over all configurations for participant
    target_blocks = []
    for slot, config in enumerate(participant_to_configs[p]):
        
        # get relevant test items
        targets = config_sampled[config].copy()

        targets["Group"] = f"p{p}" # Group is p0 to p66 and will automatically be identified by PCIBex
        targets["config"] = config
        targets["trial_type"] = "target"

        target_blocks.append(targets)
    

    # ========= filler =========
    # sample filler configurations from configurations that are not included in the target ones
    filler_configs = random.sample(filler_pool_configs, k = 5) # 5 different configurations for fillers

    filler_blocks = [] # should be 10 in total
    # iterate through each filler configuration
    for filler_config in filler_configs:
        # sample 2 rows from that filler config
        filler = config_sampled[filler_config].sample(n=2, random_state=SEED).copy() # 2 per config, which means 10 in total

        # overwrite sentence_bad for both rows
        filler["sentence_bad"] = filler["sentence_good"].apply(lambda s: shuffle_sent(s))

        filler["Group"] = f"p{p}"
        filler["trial_type"] = "filler"
        filler["config"] = filler_config

        filler_blocks.append(filler)
    
    # ========== randomization =======
    targets_df = pd.concat(target_blocks, ignore_index=True)
    fillers_df = pd.concat(filler_blocks, ignore_index=True)
    
    all_trials = targets_df.to_dict("records") + fillers_df.to_dict("records")
    final_trials = shuffle_until_valid(all_trials)
    
    rows.append(pd.DataFrame(final_trials))

    


full = pd.concat(rows, ignore_index=True)
full.to_csv("human/stims.csv", index=False)
