# LLM evaluation pipeline

This is a repository for evaluating Large Language Models on detecting symptoms using Doctor-Patient discussions.

## In this README :point_down:
- [Initial Setup](#inital-setup)
- [Features](#features)
- [Demonstration](#demonstration)

## Inital setup

1. Create and activate a conda environment

    To do this, start by creating the provided environment. After navigating to the root of this repository, enter in your terminal:
    ```
    conda env create -f environment.yml
    ```
    And to activate the environment:
    ```
    conda activate llm-evaluation
    ```

2. Provide your Client API key

    Rename the file **.env.sample** to **.env**

    Enter your Client API key in this file.

3. Insert your data in the **/data** folder.

    The data can be added in the form of csv files, or folders containing csv files.


## Features

This pipeline aims to be user-friendly and modular. It does so by separating the tasks in four main components:
- **API**

    Instantiating API objects allows to seemlessly operate with different APIs. If models from different APIs have to be tested, the logic 
    to handle the different client behaviours can be implemented here.
    
- **Dataloader**

    Dataloader handles all the data preprocessing steps. It can work with a variety of inputs, runs sanity checks on its own and provides 
    the user with a standardized dataframe that simplifies downstream operations.

- **Prompter**

    Prompter objects generates model outputs by handling model-specific prompting methods and output structuring logic. It works with binary 
    and multilabel classification tasks. To efficiently generate predictions for large inputs, it integrates parallelized prediction generation 
    and token usage tracking.

- **Scorer**

    Scorer is used to assess the performance of prompter models. It works with binary and multilabel classification tasks, providing metrics and 
    plots for the user. To build confidence intervals without spending additional tokens, it uses bootstrapping.


## Demonstration

For a demonstration of the pipeline's usage, please refer to [this notebook](scripts/demo.ipynb).


#TODO:
- write README
- implement multilabel
- rewrite demo
- write report
- test a few more queries