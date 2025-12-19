import mesa
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pprint
from io import StringIO
from datetime import datetime
import networkx as nx
import ast
from collections import defaultdict
from mesa import Agent, Model
from mesa.time import RandomActivation  # random order of agent actions
from mesa.space import MultiGrid  # multiple agents per cell
from mesa.datacollection import DataCollector
from model import run_care_model

def prepare_visualization_data(results):
    """
    Prepare the data for visualization by dropping rows before the warming-up step,
    adding a 'model_week' column starting from 1 after the warming-up period,
    and adding a 'year' column based on the model week.

    Args:
        results (dict): The results dictionary returned by the `run_care_model` function.

    Returns:
        pd.DataFrame: The prepared DataFrame for visualization.
    """
    # Extract the data and model from the results
    data = results['data']
    model = results['model']

    # Get the warming-up step
    # warming_up_step = model.warming_up_step + 1
    buffer_end_step = model.buffer_end_step + 1

    # Drop rows before the warming-up step
    data_after_warmup = data.loc[buffer_end_step:].copy()

    data_after_warmup.reset_index(drop=True, inplace=True)

    # Add a new column 'model_week' starting from 1
    data_after_warmup['model_week'] = range(1, len(data_after_warmup) + 1)

    # Add a 'year' column based on 'model_week'
    data_after_warmup['year'] = (data_after_warmup['model_week'] - 1) // 52 + 1

    return data_after_warmup

def aggregate_step_approaches_by_year(data):
    """
    Aggregate the step-by-step approaches data by year.

    Args:
        data (pd.DataFrame): The prepared DataFrame with a 'year' column.

    Returns:
        pd.DataFrame: A DataFrame aggregated by year with summed step-based approaches.
    """
    # Group by year and sum the step-based approaches
    yearly_data = data.groupby('year').agg({
        'step_micros_approached_randomly': 'sum',
        'step_micros_approached_recommended': 'sum',
        'step_micros_approached_coordinator': 'sum'
    }).reset_index()

    return yearly_data

def plot(data):
    # Aggregate step-by-step approaches by year
    yearly_data = aggregate_step_approaches_by_year(data)

    fig, ax = plt.subplots(4, 1, figsize=(12, 20))  # Create a 1x4 grid of subplots

    # Top plot: Provider Approaches by Type (Stacked)
    ax[0].stackplot(
        data['model_week'],
        data['number_of_micros_randomly_approached'],
        data['number_of_micros_recommended'],
        data['number_of_micros_approached_coordinator'],
        labels=['Randomly Approached', 'Recommended', 'Approached via Coordinator']
    )
    ax[0].set_title('Provider Approaches by Type')
    ax[0].set_xlabel('Model Week')
    ax[0].set_ylabel('Number of Approaches')
    ax[0].legend(loc='upper left')

    # Second plot: System Capacity and Care Delivery (Line Plot)
    data.plot(
        kind='line',
        x='model_week',
        y=['calc_micros', 'calc_is_receiving_care', 'resident_population'],
        ax=ax[1]
    )
    ax[1].set_title('System Capacity and Care Delivery')
    ax[1].set_xlabel('Model Week')
    ax[1].set_ylabel('Count')
    ax[1].legend(['Microproviders', 'Receiving Care', 'Resident Population'], loc='upper left')

    # Third plot: Weekly Averages (Line Plot)
    data.plot(
        kind='line',
        x='model_week',
        y=['avg_packages_of_care', 'avg_connected_microproviders'],
        ax=ax[2]
    )
    ax[2].set_title('Weekly Averages')
    ax[2].set_xlabel('Model Week')
    ax[2].set_ylabel('Average')
    ax[2].legend(['Avg Packages of Care', 'Avg Connected Microproviders'], loc='upper left')

    # Fourth plot: Step-Based Approaches by Year (Stacked Bar Chart)
    bar_width = 0.35
    ax[3].bar(
        yearly_data['year'] - bar_width / 2,
        yearly_data['step_micros_approached_randomly'],
        width=bar_width,
        label='Randomly Approached'
    )
    ax[3].bar(
        yearly_data['year'] - bar_width / 2,
        yearly_data['step_micros_approached_recommended'],
        width=bar_width,
        bottom=yearly_data['step_micros_approached_randomly'],
        label='Recommended'
    )
    ax[3].bar(
        yearly_data['year'] - bar_width / 2,
        yearly_data['step_micros_approached_coordinator'],
        width=bar_width,
        bottom=yearly_data['step_micros_approached_randomly'] + yearly_data['step_micros_approached_recommended'],
        label='Approached via Coordinator'
    )
    ax[3].set_title('Step-Based Approaches by Year (Stacked Bar Chart)')
    ax[3].set_xlabel('Year')
    ax[3].set_ylabel('Number of Approaches')
    ax[3].legend(loc='upper left')

    plt.tight_layout()
    return fig

params_with_coordinator = {
    "n_coordinators": 1,  # Coordinator enabled
    "random_seed": 42,
    "num_years": 5,
    "p_random_micro_join": 0.5  # 10% chance per step
}

# results_with_coordinator = run_care_model(params_with_coordinator)

# # Prepare the data for visualization
# data_with_coordinator = prepare_visualization_data(
#     results_with_coordinator)



