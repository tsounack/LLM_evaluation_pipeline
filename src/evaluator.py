import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix

def evaluate_configurations():
    #TODO: Implement
    pass

def evaluate_model(data: pd.DataFrame, results: pd.DataFrame, verbose: bool = True, log: bool = False) -> None:
    """
    Evaluates the model using the provided data and results.

    Args:
        data (pd.DataFrame): The data used to generate the results.
        results (pd.DataFrame): The generated results.
        verbose (bool, optional): Whether to print the evaluation results. Defaults to True.
        log (bool, optional): Whether to log the evaluation results. Defaults to False.

    Returns:
        None
    """
    if len(results.columns) == 1:
        accuracy, precision, recall, f1, confusion = evaluate_model_binary(data, results)
    else:
        #TODO: Implement for classification
        pass
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Confusion Matrix: {confusion}")
        #TODO: make it nicer
    if log:
        #TODO: Implement logging, log_path should be a parameter
        # log_file = "/path/to/log/file.log"
        # with open(log_file, "a") as f:
        #     f.write(f"Accuracy: {accuracy}\n")
        #     f.write(f"Precision: {precision}\n")
        #     f.write(f"Recall: {recall}\n")
        #     f.write(f"F1: {f1}\n")
        #     f.write(f"Confusion Matrix: {confusion}\n")
        pass

def evaluate_model_binary(data: pd.DataFrame, results: pd.DataFrame) -> tuple[float, float, float, float, list[list[int]]]:
    y_true = data["Target_binary"]
    y_true = y_true.astype(str).replace({"Positive": True, "Negative": False})
    y_pred = results
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, confusion

def evaluate_model_classification(data: pd.DataFrame, results: pd.DataFrame) -> tuple[float, float, float, float, list[np.ndarray]]:
    y_true = data["Target_classification"]
    #TODO: y_true is currently just one column, modify to multiple columns
    y_pred = results.astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    confusion = multilabel_confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, confusion