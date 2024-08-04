from .LLMAgent import LLMAgent
from .tools.drugbank import retrieval_drugbank, get_SMILES
from .tools.hetionet import retrieval_hetionet

from .utils import LOGGER

class EfficacyAgent(LLMAgent):
    def __init__(self, depth=1):
        self.name = "efficacy agent"
        self.role = ''' 
As an efficacy expert, you have the capability to assess a drug's efficacy against diseases by examining its effectiveness on the disease. 
'''

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieval_drugbank",
                    "description": "Given a drug's name, the model retrieves its information from DrugBank database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "The drug name",
                            }
                        },
                        "required": ["drug_name"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieval_hetionet",
                    "description": '''
                    Given the names of a drug and a disease, the model retrieves the path connecting the drug to the disease from the Hetionet Knowledge Graph. 
                    Hetionet is a comprehensive knowledge graph that integrates diverse biological information by connecting genes, diseases, compounds, and more into an interoperable framework. 
                    It structures real-world biomedical data into a network, facilitating advanced analysis and discovery of new insights into disease mechanisms, drug repurposing, and the genetic underpinnings of health and disease.
                    ''',
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "The drug name",
                            },
                            "disease_name": {
                                "type": "string",
                                "description": "The disease name",
                            }
                        },
                        "required": ["drug_name", "disease_name"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_SMILES",
                    "description": "Given a drug's name, the model returns its Simplified Molecular Input Line Entry System (SMILES) notation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "The drug name",
                            }
                        },
                        "required": ["drug_name"],
                    },
                }
            },
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def retrieval_drugbank(self, drug_name):
        result = retrieval_drugbank(drug_name)
        if result == "":
            return None
        else:
            return result

    def retrieval_hetionet(self, drug_name, disease_name):
        result = retrieval_hetionet(drug_name, disease_name)
        if result == "":
            return None
        else:
            return result

    def get_SMILES(self, drug_name):
        result = get_SMILES(drug_name)
        if result == "":
            return None
        else:
            return result
