from regex import L
from .LLMAgent import LLMAgent
from .reason_agent import decomposition
import re

from .safety_agent import SafetyAgent
from .efficacy_agent import EfficacyAgent
from .enrollment_agent import EnrollmentAgent

from .utils import LOGGER
from .safety_agent import SafetyAgent
from .enrollment_agent import EnrollmentAgent
from .efficacy_agent import EfficacyAgent

import time

class ClinicalAgent(LLMAgent):
    def __init__(self, user_prompt, depth=1):
        self.user_prompt = user_prompt

        self.name = "clinical agent"
        self.role = '''
You are an expert in clinical trials.
'''

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "safety_agent",
                    "description": "To understand the safety of the drug, consult the Safety Agent for safety information. Given drug name and disease name, return the historical failure rate. You also can ask the Safety Agent to get the risk of the drug and disease.",
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
                    "name": "efficacy_agent",
                    "description": "To assess the drug's efficacy against the diseases, ask the Efficacy Agent for information regarding the drug's effectiveness on the disease. Given drug name and disease name, return the drug introduction, disease introduction, and the path between drug and disease in the hetionet knowledge graph etc.",
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
                    "name": "enrollment_agent",
                    "description": "To determine if the clinical trial's eligibility criteria facilitate easy enrollment with enough patients. Given eligibility criteria, return the clinical trial will be poor enrollment, good enrollment, or excellent enrollment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eligibility_criteria": {
                                "type": "string",
                                "description": "eligibility criteria, contains including criteria and excluding criteria",
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
            },
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def safety_agent(self, drug_name, disease_name):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"Safety Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(f"[Thought] Least to Most Reasoning: Decompose the original problem", depth=1)

        safety_agent_ins = SafetyAgent(depth=2)

        decomposed_resp = decomposition(f"How can I evaluate the safety of the drug {drug_name} and disease {disease_name}?", tools=safety_agent_ins.tools)

        subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", decomposed_resp)
        subproblems = [subproblem.strip() for subproblem in subproblems]

        for idx, subproblem in enumerate(subproblems):
            LOGGER.log_with_depth(f"<subproblem>{subproblem}</subproblem>", depth=1)

        LOGGER.log_with_depth(f"[Action] Solve each subproblem...", depth=1)        
        problem_results = []
        for sub_problem in subproblems:
            LOGGER.log_with_depth(f"Solving...", depth=1)
            response = safety_agent_ins.request(f"The original user problem is: {self.user_prompt}\nNow, please you solve this problem: {sub_problem}")

            if response == "":
                LOGGER.log_with_depth(f"<solution>No solution found</solution>", depth=1)
                problem_results.append("No solution found")
            else:
                LOGGER.log_with_depth(f"<solution>{response}</solution>", depth=1)
                problem_results.append(response)

        return '\n'.join(problem_results)


    def enrollment_agent(self, eligibility_criteria, drug_name, disease_name):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"Enrollment Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(f"[Thought] Least to Most Reasoning: Decompose the original problem", depth=1)

        enrollment_agent_ins = EnrollmentAgent(depth=2)

        response = enrollment_agent_ins.request(f"The original user problem is: {self.user_prompt}\nNow, please you evaluate the enrollment difficulty of the clinical trial with eligibility criteria: {eligibility_criteria}, drugs: {drug_name}, diseases: {disease_name}")
        
        if response == "":
            LOGGER.log_with_depth(f"<solution>No solution found</solution>", depth=1)
            return "No solution found"
        else:
            LOGGER.log_with_depth(f"<solution>{response}</solution", depth=1)
            return response



    def efficacy_agent(self, drug_name, disease_name):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"Efficacy Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(f"[Thought] Least to Most Reasoning: Decompose the original problem", depth=1)

        efficacy_agent_ins = EfficacyAgent(depth=2)

        decomposed_resp = decomposition(f"How can I evaluate the efficacy of the drug {drug_name} on the disease {disease_name}?", tools=efficacy_agent_ins.tools)

        subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", decomposed_resp)
        subproblems = [subproblem.strip() for subproblem in subproblems]
        
        for idx, subproblem in enumerate(subproblems):
            LOGGER.log_with_depth(f"<subproblem>{subproblem}</subproblem>", depth=1)

        LOGGER.log_with_depth(f"[Action] Solve each subproblem...", depth=1)     

        problem_results = []
        for sub_problem in subproblems:
            response = efficacy_agent_ins.request(f"The original user problem is: {self.user_prompt}\nNow, please you solve this problem: {sub_problem}")
            
            if response == "":
                LOGGER.log_with_depth(f"<solution>No solution found</solution>", depth=1)
                problem_results.append("No solution found")
            else:
                LOGGER.log_with_depth(f"<solution>{response}</solution>", depth=1)
                problem_results.append(response)
            
        return '\n'.join(problem_results)


# if __name__ == "__main__":
#     user_problem = open("demo_prompt.txt", "r").read()
#     LOGGER.log_with_depth(user_problem)

#     safety_agent_results = safety_agent("dasatinib", user_problem)
#     LOGGER.log_with_depth(safety_agent_results)

#     enrollment_agent_results = enrollment_agent("Inclusion Criteria: Patients with type 2 diabetes; Exclusion Criteria: Patients with type 1 diabetes", user_problem)
#     LOGGER.log_with_depth(enrollment_agent_results)

#     efficacy_agent_results = efficacy_agent("dasatinib", "chronic myeloid leukemia", user_problem)
#     LOGGER.log_with_depth(efficacy_agent_results)