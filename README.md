# LLM evaluation pipeline

This is a repository for evaluating Large Language Models on detecting symptoms using Doctor-Patient discussions.

## In this README :point_down:
- [Initial Setup](#inital-setup)
- [Features](#features)

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


## Features

There are four main components to this pipeline:
- API
- Dataloader
- Prompter
- Scorer


#TODO:
- write README
- implement multilabel
- rewrite demo
- write report
- test a few more queries