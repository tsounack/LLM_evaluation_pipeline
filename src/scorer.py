import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
from typing import Union
from tqdm import tqdm
pd.set_option('future.no_silent_downcasting', True)

def scorer_factory(scorer_type: str, data: pd.DataFrame, results: pd.DataFrame) -> Union['BinaryScorer', 'MultilabelScorer']:
    """
    Factory function to create scorer objects based on the given scorer_type.

    Args:
        scorer_type (str): The type of scorer to create. Valid values are 'binary' and 'multilabel'.
        data (pd.DataFrame): The data used to generate the results.
        results (pd.DataFrame): The generated results.

    Returns:
        Scorer: An instance of the appropriate scorer type.

    Raises:
        ValueError: If an invalid scorer type is provided.
    """
    if scorer_type == "binary":
        return BinaryScorer(data, results)
    elif scorer_type == "multilabel":
        return MultilabelScorer(data, results)
    else:
        raise ValueError('Invalid scorer type. Must be either "binary" or "multilabel".')

class Scorer:
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame):
        self.data = data
        self.results = results
        data = data.reset_index(drop=True)
        results = results.reset_index(drop=True)
        self.df_combined = pd.concat([self.data, self.results], axis=1)
        
    def _bootstrap_results(self, sample_size: int, n_samples = 1000) -> dict:
        metrics = ["accuracy", "precision", "recall", "f1"]
        bootstrap_results = {metric: [] for metric in metrics}
        for _ in tqdm(range(n_samples), desc="Bootstrapping"):
            sample = self.df_combined.sample(sample_size, replace=True)
            evaluation_results = self.evaluate(sample, sample.iloc[:, -len(self.results.columns):])
            for metric in metrics:
                bootstrap_results[metric].append(evaluation_results[metric])
        return {metric: pd.Series(values) for metric, values in bootstrap_results.items()}
    
    def display_bootstrap_results(self, sample_size: int, output_type: str = "text", n_samples = 1000) -> None:
        bootstrap_results = self._bootstrap_results(sample_size, n_samples)
        if output_type == "text" or output_type == "both":
            for metric, values in bootstrap_results.items():
                print(f"{metric}: {round(np.mean(values), 4)} ({round(values.quantile(q=0.025), 4)}-{round(values.quantile(0.975), 4)} 95% CI)")
        if output_type == "plot" or output_type == "both":
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            for i, (metric, values) in enumerate(bootstrap_results.items()):
                ax = axes[i // 2, i % 2]
                ax.hist(bootstrap_results[metric], bins=30, color='skyblue', edgecolor='black', density=True)
                ax.set_title(metric.capitalize())
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.axvline(x=values.quantile(q=0.025), color = 'black', ls = '--', label="95% CI" if i == 0 else None)
                ax.axvline(x=values.quantile(q=0.975), color = 'black', ls = '--')
                ax.axvline(x=np.mean(values), color='red', linestyle='-', label="Mean" if i == 0 else None)
            fig.suptitle(f"Bootstrapped Evaluation Results (nb samples={n_samples})")
            fig.legend()
            plt.tight_layout()
            plt.show()

class BinaryScorer(Scorer):
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame):
        super().__init__(data, results)

    def get_error_dataframe(self) -> pd.DataFrame:
        return self.df_combined[self.df_combined["Target binary"] != self.df_combined[self.results.columns[0]]]
    
    def evaluate(self, data: pd.DataFrame, results: pd.DataFrame) -> None:
        y_true, y_pred = data["Target binary"], results
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": confusion}
    
    def display_length_distribution(self, model_name:str) -> None:
        correct = self.df_combined[self.df_combined["Target binary"] == self.df_combined["Pred status"]]
        incorrect = self.df_combined[self.df_combined["Target binary"] != self.df_combined["Pred status"]]
        lengths1 = correct["Context"].str.len()
        lengths2 = incorrect["Context"].str.len()
        plt.hist(lengths1, bins=30, alpha=0.5, label="Correct Prediction", color='skyblue', edgecolor='black', density=True)
        plt.hist(lengths2, bins=30, alpha=0.5, label="Incorrect Prediction", color='coral', edgecolor='black', density=True)
        plt.axvline(x=np.mean(self.df_combined["Context"].str.len()), color='red', linestyle='-', label="Mean")
        plt.xlabel("Length of Context")
        plt.ylabel("Density")
        plt.title(f"Distribution of Context Length: {model_name}")
        plt.legend()
        plt.show()

class MultilabelScorer(Scorer):
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame):
        super().__init__(data, results)

    def evaluate(self) -> None:
        raise NotImplementedError
        #TODO: Implement for multilabel

def compare_models_bootstrap(dict_scorers: dict, sample_size: int, n_samples=1000):
    # TODO: compatible multitask?
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    metrics = ["accuracy", "precision", "recall", "f1"]
    bootstrap_results = {model: scorer._bootstrap_results(sample_size=sample_size, n_samples=n_samples) for model, scorer in dict_scorers.items()}
    x_labels = list(dict_scorers.keys())
    cmap = plt.get_cmap('Pastel2', len(dict_scorers))
    for i, metric in enumerate(metrics):
        y = []
        y_lower = []
        y_upper = []
        for model in dict_scorers.keys():
            bootstrap_result = bootstrap_results[model]
            metric_mean = np.mean(bootstrap_result[metric])
            y.append(metric_mean)
            y_lower.append(metric_mean - bootstrap_result[metric].quantile(q=0.025))
            y_upper.append(bootstrap_result[metric].quantile(q=0.975) - metric_mean)
        ax = axes[i // 2, i % 2]
        ax.bar(x_labels, y, yerr=[y_lower, y_upper], capsize=5, color=[cmap(i) for i in range(len(dict_scorers))])
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Model")
        ax.set_ylabel("Value")
        ax.set_ylim(0.5, 1)
    fig.suptitle(f"Bootstrapped Comparison Results (nb samples={n_samples})")
    plt.tight_layout()
    plt.show()