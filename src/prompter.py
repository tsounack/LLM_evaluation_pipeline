import json
import time
import concurrent.futures
import numpy as np
import pandas as pd
from tools.model_structs import SymptomBinary, SymptomMultilabel
from tqdm import tqdm
from typing import Union

def run_configurations(prompter_type: str, client: object, models: list[str], df: pd.DataFrame, prompt: str) -> dict:
    """
    Run configurations for prompter using the provided list of models.

    Args:
        prompter_type (str): The type of prompter. Valid values are 'binary' and 'multilabel'.
        client (object): The client object.
        models (list[str]): List of models.
        df (pd.DataFrame): The DataFrame containing the context.
        prompt (str): The prompt given to the model for output formatting.az

    Returns:
        dict: The results dictionary, using models as keys and results dataframes as values.
    """
    results = {}
    for model in models:
        prompter = prompter_factory(prompter_type, client, model)
        generated_responses = prompter.generate(df, prompt)
        results[model] = generated_responses
    return results

def prompter_factory(prompter_type: str, client: object, model: str, temperature: float = 0) -> Union['Mistral', 'Llama']:
    """
    Factory function to create prompter objects based on the given prompter_type.

    Args:
        prompter_type (str): The type of prompter to create. Valid values are 'binary' and 'multilabel'.
        client(object): The client object.
        model (str): Link for the model, to be used by the client.
        temperature (float, optional): The temperature value for response generation. Defaults to 0.
        #TODO: update

    Returns:
        Prompter: An instance of the appropriate prompter type.

    Raises:
        ValueError: If an invalid prompter type is provided.
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
        generate_single: Generates a response given a context and a prompt, with several attempts.
        TODO: update

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

    def generate(self, df: pd.DataFrame, prompt: str, max_attempts: int = 5) -> pd.DataFrame:
        """
        Generate responses for each context in the dataframe using the given prompt in a parallelized manner.

        Args:
            df (pandas.DataFrame): The dataframe containing the contexts.
            prompt (str): The prompt given to the model for output formatting.
            max_attempts (int, optional): The maximum number of attempts to generate a response. Defaults to 5.

        Returns:
            pandas.DataFrame: A dataframe containing the generated responses. It has as many columns as the response structure.
        """
        
        rows = df["Context"].tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            responses = list(tqdm(executor.map(lambda c: self.generate_single(prompt, c, max_attempts), rows),
                                  desc=f"{self.prompter_type} task using: {self.model}",
                                  total=len(rows)))
        return pd.DataFrame(responses)

    def generate_single(self, context: str, prompt: str, max_attempts: int = 5) -> dict:
        """
        Generates a response given a context and a prompt, with several attempts.

        Args:
            context (str): The context - Doctor / Patient conversation.
            prompt (str): The prompt given to the model for output formatting.
            max_attempts (int, optional): The maximum number of attempts to generate a response. Defaults to 5.

        Returns:
            dict: The generated response as a dictionary.

        """
        attempt = 0
        while attempt < max_attempts:
            try:
                attempt += 1
                output = self._generate(context, prompt)
                self.symptom_struct(**output) # Will go to exception if cannot unpack
                return output
            except Exception as e:
                if attempt < max_attempts:
                    sleep_time = 2 ** (attempt - 2)  # Exponential backoff formula
                    time.sleep(sleep_time)
                else:
                    return np.nan

    def _generate(self, context: str, prompt: str) -> dict:
        """
        Generates a response given a context and a prompt.

        Args:
            context (str): The context - Doctor / Patient conversation.
            prompt (str): The prompt given to the model for output formatting.

        Returns:
            dict: The generated response as a dictionary.

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
        output=json.loads(output)
        return output
    
    def _set_prompter_type(self, prompter_type: str) -> None:
        if prompter_type == "binary":
            self.symptom_struct = SymptomBinary
            self.prompter_type = "binary"
        elif prompter_type == "multilabel":
            self.symptom_struct = SymptomMultilabel
            self.prompter_type = "multilabel"
        else:
            raise ValueError("Invalid prompter type")
        with open(f"tools/{self.prompter_type}.json", "r") as file:
            self.tool = json.load(file)
        self.tool_choice = {"type": "function", "function": {"name": f"symptom_{self.prompter_type}"}}

    
class Mistral(Prompter):
    def __init__(self, prompter_type: str, client: object, model: str, temperature: float = 0) -> None:
        super().__init__(client, model, temperature)
        self._set_prompter_type(prompter_type)


class Llama(Prompter):
    def __init__(self, prompter_type: str, client: object, model: str, temperature: float = 0) -> None:
        super().__init__(client, model, temperature)
        # self._set_prompter_type(prompter_type) #TODO: is this useful?
