## Risk Model
To run the disease_risk model, you need to train the enrollment firstly.

copy `outcome2label.txt` from enrollment/data/IQVIA/ to risk_model/data/

run `__init__.py`


if you have AttributeError on Line 29 and Line 39, convert
`drug_df['drugs'] = drug_df['drugs'].str.strip().lower()`
`disease_df['diseases'] = disease_df['diseases'].str.strip().lower()`
to
```
drug_df['drugs'] = drug_df['drugs'].astype(str).str.strip().str.lower()
disease_df['diseases'] = disease_df['diseases'].astype(str).str.strip().str.lower()
```