import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree as ET
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriteriaModel(nn.Module):
    def __init__(self):
        super(CriteriaModel, self).__init__()
        self.sentence_embedding_dim = 768

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sentence_embedding_dim, nhead=2, dropout=0.2, batch_first=True, dim_feedforward=2*self.sentence_embedding_dim)
        layer_norm = nn.LayerNorm(self.sentence_embedding_dim)
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=1, norm=layer_norm)
        
        self.fc1 = nn.Linear(4*self.sentence_embedding_dim, self.sentence_embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.sentence_embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, 4, self.sentence_embedding_dim)

        x = self.transformer_encoder(x)
        x = x.reshape(-1, 4*self.sentence_embedding_dim)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def partition_criteria(criteria):
    lines = [line.strip() for line in criteria.lower().split('\n') if line.strip()]

    inclusion_criteria, exclusion_criteria = [], []
    
    # Use a flag to indicate whether we are currently reading inclusion or exclusion criteria
    reading_inclusion = False
    reading_exclusion = False
    
    for line in lines:
        # Check if the line is an inclusion or exclusion header
        if 'inclusion criteria' in line:
            reading_inclusion = True
            reading_exclusion = False
            continue
        elif 'exclusion criteria' in line:
            reading_inclusion = False
            reading_exclusion = True
            continue
        
        if reading_inclusion:
            inclusion_criteria.append(line)
        elif reading_exclusion:
            exclusion_criteria.append(line)
    
    return inclusion_criteria, exclusion_criteria


def wrapper_get_sentence_embedding():
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)

    def get_sentence_embedding(sentence):
        # Encode the input string
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Send inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the output from BioBERT
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
        
        # Obtain the embeddings for the [CLS] token
        # The [CLS] token is used in BERT-like models to represent the entire sentence
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().to('cpu')
        
        return cls_embedding

    return get_sentence_embedding

# Get the sentence embedding function
get_sentence_embedding = wrapper_get_sentence_embedding()

if os.path.exists(f"{current_file_path}/data/enrollment_model.pt"):
    model = CriteriaModel().to(device)
    model.load_state_dict(torch.load(f'{current_file_path}/data/enrollment_model.pt'))
else:
    trial_outcome_df = pd.read_csv(f'{current_file_path}/data/IQVIA_trial_outcomes.csv')

    iqvia_nctid_set = set(trial_outcome_df['studyid'])
    poor_set = set(trial_outcome_df[trial_outcome_df['trialOutcome'] == 'Terminated, Poor enrollment']['studyid'])



    if os.path.exists(f'{current_file_path}/data/trial_data.csv'):
        trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')
    else:
        with open(f'{current_file_path}/data/trials/all_xml.txt', 'r') as f:
            trials_file_list = [line.strip() for line in f]

        trial_data_list = []
        for trial_path in tqdm(trials_file_list):
            nctid = trial_path.split('/')[-1].split('.')[0]

            if nctid not in iqvia_nctid_set:
                continue

            try:
                root_xml = ET.parse(f"{current_file_path}/data/{trial_path}").getroot()
                criteria = root_xml.find('eligibility').find('criteria').find('textblock').text 
                if len(criteria) == 0:
                    continue

                interventions = [i for i in root_xml.findall('intervention')]
                drug_interventions = [i.find('intervention_name').text.lower().strip() for i in interventions if i.find('intervention_type').text=='Drug']
                if len(drug_interventions) == 0:
                    continue
                drugs = ';'.join(drug_interventions)

                conditions = [i.text.lower().strip() for i in root_xml.findall('condition')]
                if len(conditions) == 0:
                    continue
                diseases = ';'.join(conditions)

                if nctid in poor_set:
                    label = 1
                else:
                    label = 0

                trial_data_list.append((nctid, criteria, drugs, diseases, label))

            except AttributeError:
                print(f"Don't have criteria or drug or diseases for {trial_path}")
            except Exception as e:
                raise e

        trial_df = pd.DataFrame(trial_data_list, columns=['nctid', 'criteria', 'drugs', 'diseases', 'label'])
        trial_df.to_csv(f'{current_file_path}/data/trial_data.csv', index=False, sep='\t')

    trial_emb_list = []
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        nctid = trial_row['nctid']
        criteria = trial_row['criteria']
        drugs = trial_row['drugs'].split(';')
        diseases = trial_row['diseases'].split(';')

        drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug) for drug in drugs]), dim=0)
        diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease) for disease in diseases]), dim=0)
        
        inclusion_criteria, exclusion_criteria = partition_criteria(criteria)

        inclusion_criteria_emb = get_sentence_embedding('\n'.join(inclusion_criteria))
        exclusion_criteria_emb = get_sentence_embedding('\n'.join(exclusion_criteria))

        trial_emb_list.append(torch.cat((inclusion_criteria_emb, exclusion_criteria_emb, drugs_emb, diseases_emb), dim=0))

    trial_emb = torch.stack(trial_emb_list)
    torch.save(trial_emb, f'{current_file_path}/data/trial_emb.pt')

    print(trial_emb.shape)

    X_data = trial_emb
    y_data = []
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        nctid = trial_row['nctid']

        if nctid in poor_set:
            y_data.append(1)
        else:
            y_data.append(0)

    y_data = torch.tensor(y_data)

    print(f"len(X_data): {len(X_data)}")
    print(f"len(y_data): {len(y_data)}")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
    weight_for_positives = class_weights[1] 

    pos_weight = torch.tensor([weight_for_positives]).to(device)
    print(pos_weight)
        
    
    class CriteriaDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = CriteriaDataset(X_train, torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = CriteriaDataset(X_test, torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = CriteriaModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

    num_epochs = 50
    best_auc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test.to(device))
            y_pred_test = y_pred_test.cpu().numpy().flatten()

            auc_test = roc_auc_score(y_test, y_pred_test)

            if auc_test > best_auc:
                best_auc = auc_test
                print(f"Epoch {epoch}\tBest AUC: {auc_test}, saving model...")

                torch.save(model.state_dict(), f'{current_file_path}/data/enrollment_model.pt')
    # Final evaluation
    model.load_state_dict(torch.load(f'{current_file_path}/data/enrollment_model.pt'))

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test.to(device))
        y_pred_test = nn.Sigmoid()(y_pred_test).cpu().numpy().flatten()

        auc_test = roc_auc_score(y_test, y_pred_test)
        acc_test = ((y_pred_test > 0.5) == y_test.numpy()).mean()
        recall_test = ((y_pred_test > 0.5) & (y_test.numpy() == 1)).sum() / (y_test.numpy() == 1).sum()

        print(f"AUC: {auc_test}, Accuracy: {acc_test}, Recall: {recall_test}")


def get_enrollment_difficulty(criteria, drugs, diseases):
    drugs = drugs.strip().lower()
    diseases = diseases.strip().lower()

    inclusion_criteria, exclusion_criteria = partition_criteria(criteria)
    inclusion_criteria_emb = get_sentence_embedding('\n'.join(inclusion_criteria))
    exclusion_criteria_emb = get_sentence_embedding('\n'.join(exclusion_criteria))
    drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug) for drug in drugs.split(';')]), dim=0)
    diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease) for disease in diseases.split(';')]), dim=0)

    model.eval()
    with torch.no_grad():
        X = torch.cat((inclusion_criteria_emb, exclusion_criteria_emb, drugs_emb, diseases_emb), dim=0).unsqueeze(0)
        y_pred = model(X.to(device))
        y_pred = nn.Sigmoid()(y_pred).cpu().numpy().flatten()

    return round(y_pred[0], 4)

