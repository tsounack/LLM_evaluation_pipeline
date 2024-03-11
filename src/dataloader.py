import os
import re
import glob
import pandas as pd
from typing import Union

ALLOWED_SYMPTOMS = ['anxiety', 'concentration problems', 'constipation', 'cough',
                    'diarrhea', 'fatigue', 'fever', 'headache', 'nausea', 
                    'numbness and tingling', 'pain', 'poor appetite', 'rash', 
                    'shortness of breath', 'trouble drinking fluids', 'vomiting', 'other']

class DataLoader:
    """
    A class for loading data from CSV files.

    Attributes:
        path (list[str]): The list of csv files

    Raises:
        FileNotFoundError: If a specified file or folder does not exist.
        ValueError: If the specified symptoms are not valid.

    Methods:
        list_csv_files: Returns a list of CSV files in the specified path(s).
        get_standardized_dataframe: Returns a standardized dataframe with specified columns for context and target(s).
        check_symptoms_validity: Checks if the dataframe's symptoms are valid.
    """

    def __init__(self, path: Union[str, list[str]]) -> None:
        """
        Initialize a DataLoader object.

        Args:
            path (Union[str, list[str]]): The path or list of paths to the CSV files or the folders containing them.

        Returns:
            None
        """
        self.path = path
        if isinstance(self.path, str):
            self.path = [self.path]
        # Check if the specified file(s) or folder(s) exist
        self._check_existence()

    def list_csv_files(self) -> list[str]:
        """
        Returns a list of CSV files in the specified path(s).

        Returns:
            list[str]: A list of CSV file paths.
        """
        files = []
        for p in self.path:
            if os.path.isdir(p):
                # Use glob to get a list of files with the .csv extension
                local_files = glob.glob(os.path.join(p, "*.csv"))
                # Custom sorting based on the numeric part of the filenames
                local_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
                files.extend(local_files)
            else:
                files.append(p)
        return files
    
    def get_standardized_dataframe(self,
                                   context_col: str = "Text Data",
                                   target_binary_col: str = "symptom_status_gs",
                                   target_multilabel_col: str = "symptom_detail_gs",
                                   keep_other_cols: bool = True,
                                   allowed_symptoms: list[str] = ALLOWED_SYMPTOMS) -> pd.DataFrame:
        """
        Returns a standardized dataframe with specified columns for context and target(s).

        Args:
            context_col (str): The name of the column containing the context data. Defaults to "Text Data".
            target_binary_col (str): The name of the column containing the binary target data. Defaults to "symptom_status_gs".
            target_multilabel_col (str): The name of the column containing the multilabel target data. Defaults to "symptom_detail_gs".
            keep_other_cols (bool): Whether to keep other columns in the dataframe. Defaults to True.
            allowed_symptoms (list[str]): The list of allowed symptoms. Defaults to ALLOWED_SYMPTOMS.

        Returns:
            pd.DataFrame: The standardized dataframe.
        """
        dataframe = self._get_dataframe()
        dataframe.rename(columns={context_col: "Context"}, inplace=True)
        # Rename binary target column, convert it to boolean
        if target_binary_col in dataframe.columns:
            dataframe.rename(columns={target_binary_col: "Target binary"}, inplace=True)
            dataframe["Target binary"] = dataframe["Target binary"].replace({"Positive": True, "Negative": False}).astype(bool)
        # Create multilabel target columns for each possible symptom, convert them to boolean
        if target_multilabel_col in dataframe.columns:
            for symptom in allowed_symptoms:
                dataframe[f"Target {symptom}"] = dataframe[target_multilabel_col].apply(
                    lambda x: symptom in str(x).split(";") if pd.notnull(x) else False
                    ).astype(bool)
        # Keep only the specified columns if keep_other_cols is False
        if not keep_other_cols:
            cols_to_keep = ["Context"]
            if "Target binary" in dataframe.columns:
                cols_to_keep.append("Target binary")
            if "Target multilabel" in dataframe.columns:
                for symptom in allowed_symptoms:
                    cols_to_keep.append(f"Target {symptom}")
            dataframe = dataframe[cols_to_keep]
        return dataframe
    
    def check_symptoms_validity(self, allowed_symptoms: list[str], symptoms_col: str = "symptom_detail_gs") -> None:
        """
        Checks if the dataframe's symptoms are valid.

        Args:
            allowed_symptoms (list[str]): The list of allowed symptoms.
            symptoms_col (str): The name of the column containing the symptoms data. Defaults to "symptom_detail_gs".

        Raises:
            ValueError: If the specified symptoms are not valid.

        Returns:
            None
        """
        dataframe = self._get_dataframe()
        symptoms = dataframe[symptoms_col].unique()
        # Split the strings, flatten the list of lists, and remove duplicates
        symptoms_list = list(set(symptom for row in symptoms if isinstance(row, str) for symptom in row.split(';')))
        # Assuming symptoms is a list of symptom strings
        cleaned_symptoms = [symptom.strip().lower() for symptom in symptoms_list]
        invalid_symptoms = [symptom for symptom in cleaned_symptoms if symptom not in allowed_symptoms]
        if invalid_symptoms:
            raise ValueError(f"Symptom list contains invalid symptoms: {invalid_symptoms}")
        else:
            print("Symptoms in dataframe are valid.")

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame by reading and concatenating the CSV files.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the CSV files.
        """
        files = self.list_csv_files()
        dataframes = []
        for file in files:
            dataframes.append(pd.read_csv(file))
        return pd.concat(dataframes, ignore_index=True)
    
    def _check_existence(self) -> None:
        """
        Checks if the specified file or folder exists.

        Raises:
            FileNotFoundError: If a specified file or folder does not exist.
        
        Returns:
            None
        """
        for p in self.path:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File/folder {p} does not exist")
            
