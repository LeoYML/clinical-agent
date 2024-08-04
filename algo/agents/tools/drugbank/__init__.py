import pandas as pd
import os
import sys

cwd_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{cwd_path}/../../')

from utils import match_name


drugbank_df = pd.read_csv(f"{cwd_path}/data/drugbank.csv", sep='\t')
drug_names = drugbank_df['name'].str.lower().tolist()

def retrieval_drugbank(drug_name):
    drug_name = drug_name.strip().lower()
    drug_name = match_name(drug_name, drug_names)

    db_row = drugbank_df[drugbank_df['name'] == drug_name]

    if db_row.empty:
        return ""
    
    drug_name = db_row['name'].values[0]
    drug_description = db_row['description'].values[0]
    drug_indication = db_row['indication'].values[0]
    drug_smiles = db_row['smiles'].values[0]
    drug_absorption = db_row['absorption'].values[0]
    drug_distribution = db_row['distribution'].values[0]
    drug_metabolism = db_row['metabolism'].values[0]
    drug_excretion = db_row['excretion'].values[0]
    drug_toxicity = db_row['toxicity'].values[0]

    drugbank_info = f''' 
    <drug name>{drug_name}</drug name>,
    <drug description>{drug_description}</drug description>,
    <drug pharmacology indication>{drug_indication}</drug pharmacology indication>,
    <drug absorption>{drug_absorption}</drug absorption>,
    <drug volume-of-distribution>{drug_distribution}</drug volume-of-distribution>,
    <drug metabolism>{drug_metabolism}</drug metabolism>,
    <drug route-of-elimination>{drug_excretion}</drug route-of-elimination>,
    <drug toxicity>{drug_toxicity}</drug toxicity>
    '''

    return drugbank_info

def get_SMILES(drug_name):
    drug_name = drug_name.strip().lower()

    drug_name = match_name(drug_name, drug_names)

    db_row = drugbank_df[drugbank_df['name'] == drug_name]

    if db_row.empty:
        return ""

    return db_row['smiles'].values[0]


if __name__ == "__main__":
    # LOGGER.log_with_depth(retrieval_drugbank("Dasatinib"))

    print(get_SMILES("Dasatinib"))
