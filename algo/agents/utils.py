from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import json
import Levenshtein
import os
import logging
import sys
import inspect

load_dotenv()
client = OpenAI()

# [gpt-3.5-turbo, gpt-4-turbo]
GPT_MODEL = 'gpt-3.5-turbo'


# Custom Logger class
class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level=level)

        self.init_stack = len(inspect.stack())

    def log_with_depth(self, msg, depth=None):
        """
        Log a message considering the call depth.
        """
        if depth is None:
            depth = self._get_call_depth()

        # Customizing the log message based on depth
        prefix = self._get_prefix_by_depth(depth)
        msg = f"{prefix}{msg}"
        self.log(logging.INFO, msg)

    def _get_call_depth(self):
        """
        Calculate the call stack depth.
        """
        stack = inspect.stack()

        # Adjust the calculation as needed
        return len(stack) - self.init_stack - 1

    def _get_prefix_by_depth(self, depth):
        """
        Create a prefix string of dashes based on the call depth.
        """
        # Define how many dashes per depth level
        dashes_per_depth = 8
        total_dashes = dashes_per_depth * depth

        return '-' * total_dashes

def setup_custom_logger(name):
    logger = CustomLogger(name)
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
    console_handler.setFormatter(logging.Formatter(
        '%(message)s'))  # Simplified format

    return logger


LOGGER = setup_custom_logger("my_logger")


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(1))
def llm_request(messages, tools=None, model=GPT_MODEL):
    try:
        if tools and len(tools) > 0:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )

        return response
    except Exception as e:
        try:
            if tools and len(tools) > 0:
                response = client.chat.completions.create(
                    model='gpt-4-turbo',
                    messages=messages,
                    tools=tools
                )
            else:
                response = client.chat.completions.create(
                    model='gpt-4-turbo',
                    messages=messages
                )

            return response
        except Exception as e:
            LOGGER.log_with_depth("Unable to generate ChatCompletion response")
            LOGGER.log_with_depth(messages)
            LOGGER.log_with_depth(tools)
            LOGGER.log_with_depth(f"Exception: {e}")
            raise e


def exec_func(response_choice):
    results = []

    if response_choice.finish_reason == "tool_calls":
        tool_calls = response_choice.message.tool_calls
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                arguments = json.loads(arguments)

                result = eval(f"{function_name}(**{arguments})")

                if result is None:
                    results.append(
                        f"The function {function_name} is called and the result is None")
                else:
                    results.append(
                        f"The function {function_name} is called and the result is: {result}")
            except Exception as e:
                LOGGER.log_with_depth(function_name)
                LOGGER.log_with_depth(arguments)
                raise Exception(f"Error executing function: {e}")

    return results


def find_least_levenshtein_distance(target_string, array):
    array = list(array)

    # Ensure the array is not empty
    if not array:
        return None, float('inf')

    # Initialize minimum distance and corresponding string
    min_distance = float('inf')
    min_string = None

    # Iterate through each string in the array
    for string in array:
        # Calculate the Levenshtein distance
        distance = Levenshtein.distance(target_string, string)

        if distance == 0:
            LOGGER.log_with_depth(f"Similar Name: {target_string} -> {string}, levenshtein distance: {distance}", depth=2)
            return string, 0

        # Update minimum distance and string if a new minimum is found
        if distance < min_distance:
            min_distance = distance
            min_string = string
    
    LOGGER.log_with_depth(f"Similar Name: {target_string} -> {min_string}, levenshtein distance: {min_distance}", depth=2)

    return min_string, min_distance


cwd_path = os.path.dirname(os.path.abspath(__file__))
name_synonyms = json.load(open(f"{cwd_path}/tools/drugbank/data/name_synonyms.json", 'r'))


def get_drug_synonyms(drug_name):
    drug_name = drug_name.strip().lower()
    if drug_name in name_synonyms:
        return name_synonyms[drug_name]
    else:
        return [drug_name]


def match_name(name, target_all_names):
    name_synonyms = get_drug_synonyms(name)
    for n in name_synonyms:
        if n in target_all_names:
            return n
        
    LOGGER.log_with_depth(f"Name: {name} and its synonyms not found")
    LOGGER.log_with_depth(f"Similary Name Matching...")

    similar_name, distance = find_least_levenshtein_distance(
        name, target_all_names)

    return similar_name
