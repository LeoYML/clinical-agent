import pandas as pd
import os
import json
import sys

cwd_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{cwd_path}/../../')

from utils import match_name, LOGGER

if os.path.exists(f"{cwd_path}/data/drug_success_ratio.json") and os.path.exists(f"{cwd_path}/data/disease_success_ratio.json"):
    drug_success_ratio = json.load(open(f"{cwd_path}/data/drug_success_ratio.json", 'r'))
    disease_success_ratio = json.load(open(f"{cwd_path}/data/disease_success_ratio.json", 'r'))
else:
    # NCT ID to label
    trial_outcome_df = pd.read_csv(f'{cwd_path}/../enrollment/data/IQVIA_trial_outcomes.csv')
    outcome2label = pd.read_csv(f'data/outcome2label.txt', sep='\t', header=None).set_index(0)[1].to_dict()
    trial_outcome_df['label'] = trial_outcome_df['trialOutcome'].map(outcome2label)
    nctid2label_df = trial_outcome_df[trial_outcome_df['label'].isin([0, 1])][['studyid', 'label']]

    # Merge with trial data
    trial_df = pd.read_csv(f'{cwd_path}/../enrollment/data/trial_data.csv', sep='\t').drop('label', axis=1)
    trial_success_df = pd.merge(trial_df, nctid2label_df, left_on='nctid', right_on='studyid', how='inner').drop('studyid', axis=1)
    trial_success_df.to_csv(f"{cwd_path}/data/trial_success.csv", sep='\t', index=False)

    drug_df = trial_success_df[['drugs', 'label']]
    drug_df['drugs'] = drug_df['drugs'].str.strip().lower()
    drug_df['drugs'] = drug_df['drugs'].str.split(';')
    drug_df_expanded = drug_df.explode('drugs').reset_index(drop=True)
    drug_df_expanded['drugs'] = drug_df_expanded['drugs'].str.strip()

    # Calculate the label ratio for every drug
    drug_label_ratio = drug_df_expanded.groupby('drugs')['label'].mean()
    drug_success_ratio = drug_label_ratio.to_dict()

    json.dump(drug_success_ratio, open(f"{cwd_path}/data/drug_success_ratio.json", 'w'))

    disease_df = trial_success_df[['diseases', 'label']]
    disease_df['diseases'] = disease_df['diseases'].str.strip().lower()
    disease_df['diseases'] = disease_df['diseases'].str.split(';')
    disease_df_expanded = disease_df.explode('diseases').reset_index(drop=True)
    disease_df_expanded['diseases'] = disease_df_expanded['diseases'].str.strip()

    # Calculate the label ratio for every disease
    disease_label_ratio = disease_df_expanded.groupby('diseases')['label'].mean()
    disease_success_ratio = disease_label_ratio.to_dict()

    json.dump(disease_success_ratio, open(f"{cwd_path}/data/disease_success_ratio.json", 'w'))

def get_disease_risk(disease_name):
    disease_name = disease_name.strip().lower()

    if disease_name in disease_success_ratio:
        return round(1 - disease_success_ratio[disease_name], 4)
    else:
        return None

def get_drug_risk(drug_name):
    drug_name = drug_name.strip().lower()
    drug_name = match_name(drug_name, drug_success_ratio.keys())

    if drug_name in drug_success_ratio:
        return round(1 - drug_success_ratio[drug_name], 4)
    else:
        return None
