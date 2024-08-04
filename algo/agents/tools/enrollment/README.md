## ClinicalTrial.gov
For more information and resources related to HetioNet, visit the:
[ClinicalTrial.gov](https://clinicaltrials.gov)

### Getting Started
1. **Download the ClinicalTrial Data**  
```
    cd data
    wget https://clinicaltrials.gov/AllPublicXML.zip
```
2. **Decompress the Data File**  
```
    unzip AllPublicXML.zip -d trials
    find trials/* -name "NCT*.xml" | sort > trials/all_xml.txt
```
3. **IQVIA Label Data**
Download the IQVIA label to data/:
https://github.com/futianfan/clinical-trial-outcome-prediction/tree/main/IQVIA

Rename the file name trial_outcomes_v1.csv to IQVIA_trial_outcomes.csv.

### Initial Setup
- **Automated Model Training**  
  The first time you call the enrollment model, the system will automatically train the model. This process takes approximately 3 hours.
  
- **Manual Graph Creation**  
  Alternatively, you can manually train the model by running `python __init__.py` in the command line.

### Model Performance
AUC: 0.7037358550062651, Accuracy: 0.7689431704885344, Recall: 0.4483221476510067