from .LLMAgent import LLMAgent
from .tools.risk_model import get_disease_risk, get_drug_risk
from .tools.drugbank import retrieval_drugbank, get_SMILES

from .utils import LOGGER

class SafetyAgent(LLMAgent):
    def __init__(self, depth=1):
        self.name = "safety agent"
        self.role = ''' 
As a drug safety expert, you have the capability to assess a drug's and disease's safety by get the drug and disease historical failure rate. 
'''
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_disease_risk",
                    "description": "Given the name of a disease, the model returns the historical failure rate of the target disease in clinical trials. The model will return a value between 0 and 1, where a higher value indicates a greater risk of the disease.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "disease_name": {
                                "type": "string",
                                "description": "The disease name",
                            }
                        },
                        "required": ["disease_name"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_drug_risk",
                    "description": "Given the name of a drug, the model returns the historical failure rate of the target disease in clinical trials. The model will return a value between 0 and 1, where a higher value indicates a greater risk of the drug.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "The drug name",
                            }
                        },
                        "required": ["drug_name"],
                    }
                }
            },
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def get_disease_risk(self, disease_name):
        result = get_disease_risk(disease_name)

        return f"The historical failure rate of {disease_name} in clinical trials is {result}."

    def get_drug_risk(self, drug_name):
        result = get_drug_risk(drug_name)

        return f"The historical failure rate of {drug_name} in clinical trials is {result}."
