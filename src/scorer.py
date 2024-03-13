import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
from typing import Union
from tqdm import tqdm
pd.set_option('future.no_silent_downcasting', True)
METRICS = ["accuracy", "precision", "recall", "f1", "Unstructured output ratio"]


def scorer_factory(scorer_type: str, data: pd.DataFrame, results: pd.DataFrame, model_name: str) -> Union['BinaryScorer', 'MultilabelScorer']:
    """
    Factory function to create scorer objects based on the given scorer_type.

    Args:
        scorer_type (str): The type of scorer to create. Valid values are 'binary' and 'multilabel'.
        data (pd.DataFrame): The data used to generate the results.
        results (pd.DataFrame): The generated results.
        model_name (str): The name of the model being evaluated for plot legends.

    Raises:
        ValueError: If an invalid scorer type is provided.

    Returns:
        Scorer: An instance of the appropriate scorer type.
    """
    if scorer_type == "binary":
        return BinaryScorer(data, results, model_name)
    elif scorer_type == "multilabel":
        return MultilabelScorer(data, results, model_name)
    else:
        raise ValueError('Invalid scorer type. Must be either "binary" or "multilabel".')

class Scorer:
    """
    A class for scoring and evaluating models using bootstrapping.

    Attributes:
        data (pd.DataFrame): The input data for evaluation.
        results (pd.DataFrame): The model predictions (output of prompter.generate).
        model_name (str): The name of the model being evaluated for plot legends.

    Methods:
        get_error_dataframe: Returns a dataframe containing the rows with prediction errors.
        display_bootstrap_results: Displays the bootstrapped evaluation results.
        display_length_distribution: Plots the distribution of context lengths for correct and incorrect predictions.
    """

    def __init__(self, data: pd.DataFrame, results: pd.DataFrame, model_name: str) -> None:
        """
        Initializes a Scorer object.

        Args:
            data (pd.DataFrame): The input data for evaluation.
            results (pd.DataFrame): The model predictions (output of prompter.generate).

        Returns:
            None
        """
        # useful if data is a sample of the original data
        data = data.reset_index(drop=True)
        results = results.reset_index(drop=True)
        self.data = data
        self.results = results
        self.model_name = model_name
        self.df_combined = pd.concat([data, results], axis=1)
        
    def _bootstrap_results(self, sample_size: int, n_samples = 1000) -> dict:
        """
        Perform bootstrapping on the given sample size and number of samples.
        
        Args:
            sample_size (int): The size of each bootstrap sample.
            n_samples (int, optional): The number of bootstrap samples to generate. Default is 1000.
        
        Returns:
            dict: A dictionary matching each metric to its list of values (computed with bootstrap resampling)
        """
        bootstrap_results = {metric: [] for metric in METRICS}
        # Sample with replacement and store the metrics
        for _ in tqdm(range(n_samples), desc=f"Bootstrapping {self.model_name}"):
            sample = self.df_combined.sample(sample_size, replace=True)
            evaluation_results = self.evaluate(sample)
            for metric in METRICS:
                bootstrap_results[metric].append(evaluation_results[metric])
        return {metric: pd.Series(values) for metric, values in bootstrap_results.items()}
    
    def get_error_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe containing the rows with prediction errors.

        Returns:
            pd.DataFrame: The dataframe containing the rows with prediction errors.
        """
        return self.df_combined[(self.df_combined[self.target_columns].values != self.df_combined[self.pred_columns].values).any(axis=1)]
    
    def display_bootstrap_results(self, sample_size: int, output_type: str = "text", n_samples = 1000) -> None:
        """
        Display the bootstrap results - text and/or plot.

        Args:
            sample_size (int): The size of each bootstrap sample.
            output_type (str): The type of output to display. Options are "text", "plot", or "both". Default is "text".
            n_samples (int): The number of bootstrap samples to generate. Default is 1000.

        Returns:
            None
        """
        bootstrap_results = self._bootstrap_results(sample_size, n_samples)
        if output_type == "text" or output_type == "both":
            for metric, values in bootstrap_results.items():
                print(f"{metric}: {round(np.mean(values), 4)} ({round(values.quantile(q=0.025), 4)}-{round(values.quantile(0.975), 4)} 95% CI)")
        if output_type == "plot" or output_type == "both":
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
            for i, (metric, values) in enumerate(bootstrap_results.items()):
                ax = axes[i // 3, i % 3]
                ax.hist(bootstrap_results[metric], bins=30, color='skyblue', edgecolor='black', density=True)
                ax.set_title(metric.capitalize())
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.axvline(x=values.quantile(q=0.025), color = 'black', ls = '--', label="95% CI" if i == 0 else None)
                ax.axvline(x=values.quantile(q=0.975), color = 'black', ls = '--')
                ax.axvline(x=np.mean(values), color='red', linestyle='-', label="Mean" if i == 0 else None)
            fig.suptitle(f"{self.model_name} - Bootstrapped Evaluation Results (nb samples={n_samples})")
            fig.legend()
            plt.tight_layout()
            # centering the last two plots
            axes[1][2].set_visible(False)
            axes[1][0].set_position([0.24,0.125,0.228,0.343])
            axes[1][1].set_position([0.55,0.125,0.228,0.343])
            plt.show()

    def display_length_distribution(self) -> None:
        """
        Plots the distribution of context lengths for correct and incorrect predictions.

        Returns:
            None
        """
        incorrect = self.get_error_dataframe()
        correct = self.df_combined.merge(incorrect, how='left', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        correct = correct.drop(columns=['_merge'])
        incorrect_nans = incorrect[incorrect[self.pred_columns].isna().any(axis=1)]
        incorrect = incorrect[~incorrect[self.pred_columns].isna().any(axis=1)]
        lengths_correct = correct["Context"].str.len()
        lengths_incorrect = incorrect["Context"].str.len()
        lengths_format_error = incorrect_nans["Context"].str.len()
        plt.hist(lengths_correct, bins=30, alpha=0.5, label="Correct Prediction", color='skyblue', 
                 edgecolor='black')
        plt.hist(lengths_incorrect, bins=30, alpha=0.5, label="Incorrect Prediction", color='coral', 
                 edgecolor='black')
        if len(incorrect_nans):
            plt.hist(lengths_format_error, bins=30, alpha=0.5, label="Format not respected", color='green', 
                     edgecolor='black')
        plt.axvline(x=np.mean(self.df_combined["Context"].str.len()), color='red', linestyle='-', label="Mean")
        plt.xlabel("Length of Context")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Context Length: {self.model_name}")
        plt.legend()
        plt.show()


class BinaryScorer(Scorer):
    """
    A class for evaluating binary predictions and calculating performance metrics.

    Methods:
        evaluate: Evaluates the binary predictions and returns a dictionary of performance metrics.
    """

    def __init__(self, data: pd.DataFrame, results: pd.DataFrame, model_name: str) -> None:
        """
        Initialize the BinaryScorer object.

        Args:
            data (pd.DataFrame): The input data for evaluation.
            results (pd.DataFrame): The model predictions (output of prompter.generate).
            model_name (str): The name of the model being evaluated for plot legends.

        Returns:
            None
        """
        super().__init__(data, results, model_name)
        # Keep track of the columns for predictions and ground truth
        self.pred_columns = ["Target binary"]
        self.target_columns = ["Pred status"]
    
    def evaluate(self, data: pd.DataFrame) -> dict:
        """
        Evaluate the performance of a model using the provided data.

        Args:
            data (pd.DataFrame): The input data containing the true and predicted values.

        Returns:
            dict: A dictionary containing the evaluation metrics, including accuracy, precision,
                  recall, F1 score, confusion matrix, and unstructured output ratio.
        """
        # NaN values are the unstructured outputs
        df_not_nan = data[~data['Pred status'].isna()]
        df_nan = data[data['Pred status'].isna()]
        y_true, y_pred = df_not_nan["Target binary"], df_not_nan["Pred status"]
        y_pred = y_pred.astype(bool)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        unstructured_ratio = len(df_nan) / len(data)
        return {"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": confusion,
                "Unstructured output ratio": unstructured_ratio}
    

class MultilabelScorer(Scorer):
    """
    A class for evaluating multilabel models.

    Methods:
        evaluate: Evaluates the multilabel predictions and returns a dictionary of performance metrics.
    """

    def __init__(self, data: pd.DataFrame, results: pd.DataFrame, model_name: str) -> None:
        """
        Initialize the MultilabelScorer object.

        Args:
            data (pd.DataFrame): The input data for evaluation.
            results (pd.DataFrame): The model predictions (output of prompter.generate).
            model_name (str): The name of the model being evaluated for plot legends.
        
        Returns:
            None
        """
        super().__init__(data, results, model_name)
        # Keep track of the columns for predictions and ground truth
        self.pred_columns = self.results.columns[:-1]
        self.target_columns = self.data.columns[-len(self.pred_columns):]

    def evaluate(self, data: pd.DataFrame) -> dict:
        """
        Evaluate the performance of a model using the provided data, micro-averaging the results.

        Args:
            data (pd.DataFrame): The input data containing the true and predicted values.

        Returns:
            dict: A dictionary containing the micro-averaged evaluation metrics, including accuracy, 
                precision, recall, F1 score, confusion matrix, and unstructured output ratio.
        """
        # NaN values are the unstructured outputs
        df_not_nan = data[~data[self.pred_columns].isna().any(axis=1)]
        df_nan = data[data[self.pred_columns].isna().any(axis=1)]
        y_true = df_not_nan[self.target_columns].astype(bool)
        y_pred = df_not_nan[self.pred_columns].astype(bool)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        confusion = multilabel_confusion_matrix(y_true, y_pred)
        unstructured_ratio = len(df_nan) / len(data)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion,
            "Unstructured output ratio": unstructured_ratio
        }

def compare_models_bootstrap(dict_scorers: dict, sample_size: int, n_samples=1000) -> None:
    """
    Compares the performance of multiple models using bootstrap resampling. Plots the results.

    Parameters:
    - dict_scorers (dict): A dictionary containing the models and their corresponding scorers.
    - sample_size (int): The size of each bootstrap sample.
    - n_samples (int): The number of bootstrap samples to generate. Default is 1000.

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    # Bootstrap resampling for each model
    bootstrap_results = {model: scorer._bootstrap_results(sample_size=sample_size, n_samples=n_samples) for model, scorer in dict_scorers.items()}
    x_labels = list(dict_scorers.keys())
    cmap = plt.get_cmap('Pastel2', len(dict_scorers))
    for i, metric in enumerate(METRICS):
        y = []
        y_lower = []
        y_upper = []
        for model in dict_scorers.keys():
            bootstrap_result = bootstrap_results[model]
            metric_mean = np.mean(bootstrap_result[metric])
            y.append(metric_mean)
            y_lower.append(metric_mean - bootstrap_result[metric].quantile(q=0.025))
            y_upper.append(bootstrap_result[metric].quantile(q=0.975) - metric_mean)
        ax = axes[i // 3, i % 3]
        ax.bar(x_labels, y, yerr=[y_lower, y_upper], capsize=5, color=[cmap(i) for i in range(len(dict_scorers))])
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Model")
        ax.set_ylabel("Value")
        # we can set the y axis to start at 0.5 if all values are above 0.5
        if all(y[i] - y_lower[i] > 0.5 for i in range(len(y_lower))):
            ax.set_ylim(0.5, 1)
    fig.suptitle(f"Bootstrapped Comparison Results (nb samples={n_samples})")
    plt.tight_layout()
    # centering the last two plots
    axes[1][2].set_visible(False)
    axes[1][0].set_position([0.24,0.125,0.228,0.343])
    axes[1][1].set_position([0.55,0.125,0.228,0.343])
    plt.show()