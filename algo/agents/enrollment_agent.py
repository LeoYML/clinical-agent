from .LLMAgent import LLMAgent
from .tools.enrollment import get_enrollment_difficulty

from .utils import LOGGER

class EnrollmentAgent(LLMAgent):
    def __init__(self, depth=1):
        self.name = "enrollment agent"
        self.role = '''
As an enrollment expert, you have the capability to predict the enrollment difficulty of a clinical trial based on its eligibility criteria. 
'''
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_enrollment_difficulty",
                    "description": "Based on the eligibility criteria for a clinical trial, the model estimates the difficulty of enrolling participants. It returns a value between 0 and 1, where a higher value signifies greater difficulty in enrollment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eligibility_criteria": {
                                "type": "string",
                                "description": "The eligibility criteria of the clinical trial",
                            },
                            "drug_name": {
                                "type": "string",
                                "description": "The drug name",
                            },
                            "disease_name": {
                                "type": "string",
                                "description": "The disease name",
                            }
                        },
                        "required": ["eligibility_criteria", "drug_name", "disease_name"],
                    },
                }
            }
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def get_enrollment_difficulty(self, eligibility_criteria, drug_name="", disease_name=""):
        result = get_enrollment_difficulty(eligibility_criteria, drug_name, disease_name)
        result = round(result, 4)

        return f"The enrollment difficulty of the clinical trial based on the eligibility criteria is {result}. Higher values indicate greater difficulty in enrollment."
