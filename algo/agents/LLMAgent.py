from .utils import llm_request, GPT_MODEL
import json

from .utils import LOGGER
import traceback

class LLMAgent:
    def __init__(self, name, role, examples='', tools=[], model=GPT_MODEL, depth=1):
        self.name = name
        self.role = role
        self.tools = tools
        self.examples = examples
        self.model = model

        self.depth = depth

        self.system_prompt = role

        if tools and len(tools) > 0:
            func_content = json.dumps([
                {'function_name': func['function']['name'], 'description': func['function']['description']} for func in tools
            ], indent=4)

            self.system_prompt += f"\n The following tools are available for you to use: \n{func_content}."

        if len(examples) > 0:
            self.system_prompt += f"\nFor example:{examples}"

        self.messages = [{'role': 'system', 'content': self.system_prompt}]

    def request(self, prompt):
        self.messages.append({'role': 'user', 'content': prompt})
        
        response = llm_request(self.messages, self.tools, self.model)

        results = []
        for choice in response.choices:
            if choice.finish_reason in ['tool_calls', 'function_call']:
                results.append(self.exec_func(choice))
            elif choice.finish_reason == 'stop':
                results.append((choice.message.content))
            elif choice.finish_reason == 'content_filter':
                raise Exception("Content filter triggered.")
            elif choice.finish_reason == 'length':
                raise Exception("Max token length reached.")
            else:
                raise Exception(f"Unknown finish reason: {choice.finish_reason}")
        
        results = '\n'.join(results)

        self.messages.append({'role': 'assistant', 'content': results})

        return results

        
    def exec_func(self, response_choice):
        results = []

        if response_choice.finish_reason == "tool_calls":
            tool_calls = response_choice.message.tool_calls
            for tool_call in tool_calls:
                LOGGER.log_with_depth(f"[Action] Function calling...", depth=self.depth)

                # LOGGER.log_with_depth("==============  DEBUG  ================", depth=self.depth)
                # LOGGER.log_with_depth(f"Tool call: {tool_call.function.name}")
                # LOGGER.log_with_depth(f"Arguments: {tool_call.function.arguments}", depth=self.depth)
                # LOGGER.log_with_depth("=======================================", depth=self.depth)

                try:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    arguments = json.loads(arguments)

                    if function_name == 'multi_tool_use.parallel':
                        for sub_function in arguments['tool_uses']:
                            sub_function_name = sub_function['recipient_name'].split('.')[-1]
                            sub_arguments = sub_function['parameters']

                            result = eval(f"self.{sub_function_name}(**{sub_arguments})")

                            if result is None:
                                results.append(f"<function>{sub_function_name}</function><result>NONE</result>")
                            else:
                                results.append(f"<function>{sub_function_name}</function><result>{result}</result>")
                            
                            # Agent Level results
                            if self.depth <= 1:
                                LOGGER.log_with_depth(f"<function>{sub_function_name}</function><result>{result}</result>", depth=self.depth)
                    else:
                        result =  eval(f"self.{function_name}(**{arguments})")

                        if result is None:
                            results.append(f"<function>{function_name}</function><result>NONE</result>")
                        else:
                            results.append(f"<function>{function_name}</function><result>{result}</result>")
                        
                        # Agent Level results
                        if self.depth <= 1:
                            LOGGER.log_with_depth(f"<function>{function_name}</function><result>{result}</result>", depth=self.depth)

                except AttributeError as e:
                    LOGGER.log_with_depth(f"Function name: {function_name}, Arguments: {arguments}", depth=self.depth)
                    LOGGER.log_with_depth(f"Warning: {e}", depth=self.depth)
                    traceback.print_exc()
                    results.append(f"[Function]: {function_name} is called and the result is None")
                except Exception as e:
                    LOGGER.log_with_depth(function_name, depth=self.depth)
                    LOGGER.log_with_depth(arguments, depth=self.depth)
                    traceback.print_exc()
                    raise Exception(f"Error executing function {function_name}, Arguments: {arguments}: {e}")

                break

        return '\n'.join(results)