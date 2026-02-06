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
    ax[0].set_xlabel('Year')
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

    # Annotate the second plot: System Capacity and Care Delivery (Yearly)
    for col in ['calc_micros', 'calc_is_receiving_care', 'resident_population']:
        yearly_points = data.groupby('year').first()
        for x, y in zip(yearly_points['model_week'], yearly_points[col]):
            ax[1].annotate(
                f"{y}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center', fontsize=8
            )

    # Third plot: Weekly Averages (Line Plot)
    data.plot(
        kind='line',
        x='model_week',
        y=['avg_packages_of_care', 'avg_connected_microproviders'],
        ax=ax[2]
    )
    ax[2].set_title('Weekly Averages')
    ax[2].set_xlabel('Year')
    ax[2].set_ylabel('Average')
    ax[2].legend(['Avg Packages of Care', 'Avg Connected Microproviders'], loc='upper left')

    # Annotate the third plot: Weekly Averages (Yearly)
    for col in ['avg_packages_of_care', 'avg_connected_microproviders']:
        yearly_points = round(data.groupby('year').first(), 1)
        for x, y in zip(yearly_points['model_week'], yearly_points[col]):
            ax[2].annotate(
                f"{y}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center', fontsize=8
            )
    # Map model_week to year for x-axis ticks
    year_ticks = data.groupby('year')['model_week'].first().values
    year_labels = data['year'].unique()

    # Update x-axis ticks for top 3 subplots
    for axis in ax:
        axis.set_xticks(year_ticks)
        axis.set_xticklabels(year_labels)

    #subplot 4: Step-Based Approaches by Year (Stacked Bar Chart)
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

    # Add callout values for y-axis data points on each year
    for i, row in yearly_data.iterrows():
        ax[3].annotate(
            f"{row['step_micros_approached_randomly']}",
            (row['year'] - bar_width / 2, row['step_micros_approached_randomly'] / 2),
            ha='center', va='center', color='white', fontsize=8
        )
        ax[3].annotate(
            f"{row['step_micros_approached_recommended']}",
            (row['year'] - bar_width / 2, row['step_micros_approached_randomly'] + row['step_micros_approached_recommended'] / 2),
            ha='center', va='center', color='white', fontsize=8
        )
        if row['step_micros_approached_coordinator'] > 0:
            ax[3].annotate(
                f"{row['step_micros_approached_coordinator']}",
                (row['year'] - bar_width / 2, row['step_micros_approached_randomly'] + row['step_micros_approached_recommended'] + row['step_micros_approached_coordinator'] / 2),
                ha='center', va='center', color='white', fontsize=8
        )
    
    ax[3].set_title('Step-Based Approaches by Year (Stacked Bar Chart)')
    ax[3].set_xlabel('Year')
    ax[3].set_ylabel('Number of Approaches')
    ax[3].legend(loc='upper left')
    
    ax[3].set_xticks(yearly_data['year'])
    ax[3].set_xticklabels(yearly_data['year'])

    # Restore default ticks for ax[3]
    ax[3].tick_params(axis='x', which='both', labelrotation=0)

    plt.tight_layout()
    return fig


# results_with_coordinator = run_care_model(params_with_coordinator)

# # Prepare the data for visualization
# data_with_coordinator = prepare_visualization_data(
#     results_with_coordinator)



