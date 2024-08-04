from .LLMAgent import LLMAgent
import json

from .utils import LOGGER

# Least to Most Reasoning

examples = '''
Question: How can we predict whether this clinical trial can pass?
Answer: <subproblem> To understand the drug's and disease's safety, we need to consult the Safety Agent for information on drug risks and disease risks. </subproblem>
<subproblem> To evaluate the drug's efficacy against diseases, it is essential to request information on the drug's effectiveness from the Efficacy Agent. Obtain details about the drug, including its description, pharmacological indications, absorption, volume of distribution, metabolism, route of elimination, and toxicity. </subproblem>
<subproblem> Ask the enrollment agent to assess the difficulty of enrolling enough patients. </subproblem>

Question: How can I evaluate the safety of the drug aspirin on the disease diabetes?
<subproblem> To assess the risk associated with the drug, we must use the "get_drug_risk" tool. </subproblem>
<subproblem> To evaluate the risk associated with the disease, we must use the "get_disease_risk" tool. </subproblem>
<subproblem> Provide insights into the safety of both the drug and the disease based on your expertise, without resorting to any external tools. </subproblem>

Question: How can I evaluate the efficacy of the drug aspirin on the disease diabetes?
Answer:
<subproblem> To understand the drug's structure, obtaining the SMILES notation of the drug is necessary. </subproblem>
<subproblem> Assessing the drug's effectiveness requires retrieving information from the DrugBank database. </subproblem>
<subproblem> To evaluate the drug's impact on the disease, we must obtain the pathway linking the drug to the disease from the Hetionet Knowledge Graph by using the retrieval_hetionet tool. </subproblem>
<subproblem> Offer insights on the drug and disease based on your expertise, without resorting to any external tools. </subproblem>
'''


def decomposition(original_problem, tools=None):
    name = 'decomposition agent'
    
    role = f'''
As a decomposition expert, you have the capability to break down a complex problem into smaller, more manageable subproblems. 
Utilize tools to address each subproblem individually, ensuring one tool per subproblem.
Aim to resolve every subproblem either through a specific tool or your expertise.
You don't need to solve it; your duty is merely to break down the problem into <subproblem>subproblems</subproblem>.
'''

    if tools and len(tools) > 0:
        func_content = json.dumps([
            {'function_name': func['function']['name'], 'description': func['function']['description']} for func in tools
        ], indent=4)

        role += f"\n The following tools are available for you to use: <tools>{func_content}</tools>."

    agent1 = LLMAgent(name, role, examples=examples)

    response = agent1.request(original_problem)

    return response
