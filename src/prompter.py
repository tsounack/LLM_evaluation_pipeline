import json
import time
import concurrent.futures
import os
import glob
import numpy as np
import pandas as pd
import threading
from tools.model_structs import SymptomBinary, SymptomMultilabel
from tqdm import tqdm
from typing import Union
from pathlib import Path

def prompter_factory(prompter_type: str, client: object, model: str, temperature: float = 0) -> Union['Mistral', 'Llama']:
    """
    Factory function to create prompter objects based on the given prompter_type and model name.

    Args:
        prompter_type (str): The type of prompter to create. Valid values are 'binary' and 'multilabel'.
        client(object): The client object.
        model (str): Link for the model, to be used by the client.
        temperature (float, optional): The temperature value for response generation. Defaults to 0.

    Raises:
        ValueError: If an invalid prompter model is provided.

    Returns:
        Prompter: An instance of the appropriate model type.
    """
    if "mistral" in model:
        return Mistral(prompter_type, client, model, temperature)
    elif "llama" in model:
        return Llama(prompter_type, client, model, temperature)
    else:
        raise ValueError('Invalid model type. Must be either "mistral" or "llama".')

class Prompter:
    """
    A class that generates a response from a given model using the provided context and prompt.

    Attributes:
        client (object): The client object.
        model (str): Link for the model, to be used by the client.
        temperature (float): The temperature value for response generation.

    Methods:
        generate: Generate responses for each context in the dataframe using the given prompt. Parallelized implementation.
        generate_single: Generates a response given a context and a prompt, with several attempts.
    """

    def __init__(self, client: object, model: str, temperature: float) -> None:
        """
        Initializes a Prompter object.

        Args:
            client (object): The client object.
            model (str): Link for the model, to be used by the client.
            temperature (float): The temperature value for response generation.

        Returns:
            None
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.lock = threading.Lock()
        self.num_tokens = 0

    def generate(self, df: pd.DataFrame, prompt: str, max_attempts: int = 5) -> pd.DataFrame:
        """
        Generate responses for each context in the dataframe using the given prompt. Parallelized implementation.

        Args:
            df (pandas.DataFrame): The dataframe containing the contexts.
            prompt (str): The prompt given to the model for output formatting.
            max_attempts (int, optional): The maximum number of attempts to generate a response. Defaults to 5.

        Returns:
            pandas.DataFrame: A dataframe containing the generated responses. It has as many columns as the response structure.
        """
        rows = df["Context"].tolist()
        self.num_tokens = 0 # Reset the number of tokens
        # Parallelize the generation of responses
        # potential optimization: don't start all workers at once to avoid reaching rate limit at beginning
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            responses = []
            with tqdm(total=len(df), desc=f"{self.prompter_type} task using: {self.model} - Total tokens: {self.num_tokens:,.0f}") as progress_bar:
                for output in executor.map(lambda c: self._unpack_generate_single(c, prompt, max_attempts), rows):
                    responses.append(output)
                    progress_bar.set_description(f"{self.prompter_type} task using: {self.model} - Total tokens: {self.num_tokens:,.0f}")
                    progress_bar.update(1)
        # this handles nan values in the responses (unstructured output)
        df_responses = pd.DataFrame(responses)
        # Move the column "output" to the last position. Otherwise if the first sample
        # is unstructured, the rest of the columns will be shifted to the left.
        df_responses["output"] = df_responses.pop("output")
        df_responses = df_responses.add_prefix("Pred ")  # Add Pred to every column name
        return df_responses
    
    def _unpack_generate_single(self, context: str, prompt: str, max_attempts: int = 5) -> dict:
        """
        Unpacks the output of the generate_single method.
        Since the generate_single method returns a tuple, this method unpacks the tuple and returns
            the dictionary, while keeping track of the number of tokens generated.

        Args:
            context (str): The context - Doctor / Patient conversation.
            prompt (str): The prompt given to the model for output formatting.
            max_attempts (int, optional): The maximum number of attempts to generate a response. Defaults to 5.
        
        Returns:
            dict: The generated response as a dictionary.
                For Binary: {"status": bool}
                For Multilabel: {"symptom": bool for symptom in Symptoms}
        """
        output, num_tokens = self.generate_single(context, prompt, max_attempts)
        # Wrapping in thread lock to avoid race condition when updating the number of tokens
        with self.lock:
            self.num_tokens += num_tokens
        return output

    def generate_single(self, context: str, prompt: str, max_attempts: int = 5) -> tuple[dict, int]:
        """
        Generates a response given a context and a prompt, with several attempts.

        Args:
            context (str): The context - Doctor / Patient conversation.
            prompt (str): The prompt given to the model for output formatting.
            max_attempts (int, optional): The maximum number of attempts to generate a response. Defaults to 5.

        Returns:
            dict: The generated response as a dictionary. Column output is the model's text output.
                For Binary: {"status": bool, "output": str}
                For Multilabel: {"symptom": bool for symptom in Symptoms, "output": str}
            total_tokens: The number of tokens used to generate the response.
        """
        attempt = 0
        total_tokens = 0
        output = np.nan
        # Try to generate a response with several attempts
        while attempt < max_attempts:
            try:
                attempt += 1
                structured_output, output, num_tokens = self._generate(context, prompt)
                total_tokens += num_tokens
                self.symptom_struct(**structured_output) # Will go to exception if cannot unpack
                # adding the model's text output
                structured_output["output"] = output
                return structured_output, total_tokens
            # exception might be an unstructured output or API rate limit
            # potential optimisation: formatting errors shouldn't have backoff
            except Exception as e:
                # print(e)
                if attempt < max_attempts:
                    sleep_time = 2 ** (attempt - 2)  # Exponential backoff formula
                    time.sleep(sleep_time)
                else:
                    # unstructured output. Pass the model's text output by itself
                    # (the prediction columns will contain nan for the exception of 
                    #  the output)
                    return {"output": output}, total_tokens
    
    def _set_prompter_type(self, prompter_type: str) -> None:
        """
        Sets the prompter type and initializes the corresponding symptom structure and tool.
        This is used by the Mistral class for function calling.

        Args:
            prompter_type (str): The type of prompter, either "binary" or "multilabel".

        Raises:
            ValueError: If an invalid prompter type is provided.

        Returns:
            None
        """
        if prompter_type == "binary":
            self.symptom_struct = SymptomBinary
            self.prompter_type = "binary"
        elif prompter_type == "multilabel":
            self.symptom_struct = SymptomMultilabel
            self.prompter_type = "multilabel"
        else:
            raise ValueError("Invalid prompter type")
        path = f"{Path(__file__).parent}/../tools/{self.prompter_type}.json"
        with open(path, "r") as file:
            self.tool = json.load(file)
        self.tool_choice = {"type": "function", "function": {"name": f"symptom_{self.prompter_type}"}}

    
class Mistral(Prompter):
    """
    A class representing the Mistral prompter.
    """

    def __init__(self, prompter_type: str, client: object, model: str, temperature: float = 0) -> None:
        """
        Initialize the Prompter class for Mistral models.

        Args:
            prompter_type (str): The type of prompter.
            client (object): The client object.
            model (str): The model to use.
            temperature (float, optional): The temperature value. Defaults to 0.

        Returns:
            None
        """
        super().__init__(client, model, temperature)
        self._set_prompter_type(prompter_type)
    
    def _generate(self, context: str, prompt: str) -> tuple[dict, str, int]:
        """
        Generates a response given a context and a prompt.

        Args:
            context (str): The context - Doctor / Patient conversation.
            prompt (str): The prompt given to the model for output formatting.

        Returns:
            dict: The generated response as a dictionary.
                For Binary: {"status": bool}
                For Multilabel: {"symptom": bool for symptom in Symptoms}
            str: The text output from the model - None for Mistral models since we are using function calling.
            int: The number of tokens used to generate the response.
        """
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": context}]
        completion =self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            tools=self.tool,
            tool_choice=self.tool_choice,
            messages=messages
        )
        output=completion.choices[0].message.tool_calls[0].function.arguments
        num_tokens = completion.usage.total_tokens
        output=json.loads(output)
        return output, None, num_tokens


class Llama(Prompter):
    """
    A class representing the Llama prompter.
    """

    def __init__(self, prompter_type: str, client: object, model: str, temperature: float = 0) -> None:
        """
        Initialize the Prompter class for Llama models.

        Args:
            prompter_type (str): The type of prompter.
            client (object): The client object.
            model (str): The model to use.
            temperature (float, optional): The temperature value. Defaults to 0.

        Returns:
            None
        """
        super().__init__(client, model, temperature)
        self._set_prompter_type(prompter_type) # for now only useful for labels
    
    def _generate(self, context: str, prompt: str) -> tuple[dict, str, int]:
        """
        Generates a response given a context and a prompt.

        Args:
            context (str): The context - Doctor / Patient conversation.
            prompt (str): The prompt given to the model for output formatting.

        Returns:
            dict: The generated response as a dictionary.
                For Binary: {"status": bool}
                For Multilabel: {"symptom": bool for symptom in Symptoms}
            str: The text output from the model.
            int: The number of tokens used to generate the response.
        """
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": context}]
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages
        )
        output=completion.choices[0].message.content
        num_tokens = completion.usage.total_tokens

        if self.prompter_type == "binary":
            first_word = output.split()[0].rstrip(',.')
            if first_word == 'Yes':
                processed_output = True
            elif first_word == 'No':
                processed_output = False
            else:
                processed_output = np.nan
            processed_output = {"status": processed_output}
            model_output = output

        elif self.prompter_type == "multilabel":
            processed_output = json.loads(output)
            # some outputs will add symptoms that are not in the list
            if len(processed_output) > 17:
                processed_output = {k: processed_output[k] for k in list(processed_output)[:17]}
            model_output = np.nan
        return processed_output, model_output, num_tokens