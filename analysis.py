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

def prepare_visualization_data(results_with_coord, results_without_coord):
    """
    Prepare the data for visualization by processing results from two simulations:
    one with a coordinator and one without.

    Args:
        results_with_coord (dict): Results from simulation with coordinator.
        results_without_coord (dict): Results from simulation without coordinator.

    Returns:
        tuple: Two DataFrames (data_with_coord, data_without_coord) prepared for visualization.
    """
    def preprocess_data(results):
        # Extract the data and model from the results
        data = results['data']
        model = results['model']

        # Get the warming up end step
        warmup_end_step = model.warming_up_step + 1

        # Drop rows before the warmup end step
        data_after_warmup = data.loc[warmup_end_step:].copy()
        data_after_warmup.reset_index(drop=True, inplace=True)

        # Add a new column 'model_week' starting from 1
        data_after_warmup['model_week'] = range(1, len(data_after_warmup) + 1)

        # Add a 'year' column based on 'model_week'
        data_after_warmup['year'] = (data_after_warmup['model_week'] - 1) // 52 + 1

        return data_after_warmup

    # Preprocess both datasets
    data_with_coord = preprocess_data(results_with_coord)
    data_without_coord = preprocess_data(results_without_coord)

    return data_with_coord, data_without_coord

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

def approaches_plot(data_with_coord, data_without_coord):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Plot approaches for both scenarios
    ax.plot(
        data_with_coord['model_week'],
        data_with_coord['number_of_micros_randomly_approached'] +\
        data_with_coord['number_of_micros_recommended'] +\
        data_with_coord['number_of_micros_approached_coordinator'],
        label='Model With Coordinator',
        color='blue'
        )
    ax.plot(
        data_without_coord['model_week'],
        data_without_coord['number_of_micros_randomly_approached'] +\
        data_without_coord['number_of_micros_recommended'],
        label='Model Without Coordinator',
        color='red'
        )
    ax.legend(loc='upper left')
    
    ax.set_title('Micro-Provider Approaches Over Time')

    # Map model_week to year for x-axis ticks
    year_ticks = data_with_coord.groupby('year')['model_week'].first().values
    year_labels = data_with_coord['year'].unique()

    # Update x-axis ticks for the plot
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)
    ax.set_xlabel('Model Year')
    ax.set_ylabel('Total Number of Approaches')

    return fig

def agg_approaches_plot(agg_data_with_coord, agg_data_without_coord):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Plot aggregated approaches for both scenarios
    ax.plot(
        agg_data_with_coord['year'],
        agg_data_with_coord['step_micros_approached_randomly'] +\
        agg_data_with_coord['step_micros_approached_recommended'] +\
        agg_data_with_coord['step_micros_approached_coordinator'],
        label='Model With Coordinator',
        color='blue'
        )
    ax.plot(
        agg_data_without_coord['year'],
        agg_data_without_coord['step_micros_approached_randomly'] +\
        agg_data_without_coord['step_micros_approached_recommended'],
        label='Model Without Coordinator',
        color='red'
        )
    
    year_ticks = agg_data_with_coord['year']
    year_labels = agg_data_with_coord['year']
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)
    ax.set_xlabel('Year')
    return fig

def plot_percent_receiving_care(data_with_coord, data_without_coord):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Apply rolling average to smooth the percentage receiving care
    data_with_coord['smoothed_percentage_receiving_care'] = data_with_coord['calc_receiving_care_percentage'].rolling(window=12, center=True).mean()
    data_without_coord['smoothed_percentage_receiving_care'] = data_without_coord['calc_receiving_care_percentage'].rolling(window=12, center=True).mean()

    # Plot approaches for both scenarios
    ax.plot(
        data_with_coord['model_week'],
        data_with_coord['smoothed_percentage_receiving_care'],
        label='Smoothed Model With Coordinator',
        color='blue', linestyle='--'
    )
    ax.plot(
        data_without_coord['model_week'],
        data_without_coord['smoothed_percentage_receiving_care'],
        label='Smoothed Model Without Coordinator',
        color='red', linestyle='--'
    )
    ax.set_title('Percentage of Residents Receiving Care Over Time')
    ax.set_xlabel('Model Year')
    ax.set_ylabel('Percent Receiving Care (%)')
    ax.legend(loc='upper left')

    # Map model_week to year for x-axis ticks
    year_ticks = data_with_coord.groupby('year')['model_week'].first().values
    year_labels = data_with_coord['year'].unique()

    # Update x-axis ticks for the plot
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    return fig

def plot_quality_over_time(data_with_coord, data_without_coord):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Apply rolling average to smooth the average quality
    data_with_coord['smoothed_avg_quality'] = data_with_coord['avg_micro_quality'].rolling(window=12, center=True).mean()
    data_without_coord['smoothed_avg_quality'] = data_without_coord['avg_micro_quality'].rolling(window=12, center=True).mean()

    # Plot approaches for both scenarios
    ax.plot(
        data_with_coord['model_week'],
        data_with_coord['smoothed_avg_quality'],
        label='Smoothed Model With Coordinator',
        color='blue', linestyle='--'
    )
    ax.plot(
        data_without_coord['model_week'],
        data_without_coord['smoothed_avg_quality'],
        label='Smoothed Model Without Coordinator',
        color='red', linestyle='--'
    )
    ax.set_title('Average Micro-Provider Quality Over Time')
    ax.set_xlabel('Model Year')
    ax.set_ylabel('Average Quality')
    ax.legend(loc='upper left')

    # Map model_week to year for x-axis ticks
    year_ticks = data_with_coord.groupby('year')['model_week'].first().values
    year_labels = data_with_coord['year'].unique()

    # Update x-axis ticks for the plot
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    return fig

def plot_quality_threshold_over_time(data_with_coord, data_without_coord):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Apply rolling average to smooth the average quality threshold
    data_with_coord['smoothed_avg_quality_threshold'] = data_with_coord['avg_micro_quality_threshold'].rolling(window=12, center=True).mean()
    data_without_coord['smoothed_avg_quality_threshold'] = data_without_coord['avg_micro_quality_threshold'].rolling(window=12, center=True).mean()

    # Plot approaches for both scenarios
    ax.plot(
        data_with_coord['model_week'],
        data_with_coord['smoothed_avg_quality_threshold'],
        label='Smoothed Model With Coordinator',
        color='blue', linestyle='--'
    )
    ax.plot(
        data_without_coord['model_week'],
        data_without_coord['smoothed_avg_quality_threshold'],
        label='Smoothed Model Without Coordinator',
        color='red', linestyle='--'
    )
    ax.set_title('Average Micro-Provider Quality Threshold Over Time')
    ax.set_xlabel('Model Year')
    ax.set_ylabel('Average Quality Threshold')
    ax.legend(loc='upper left')

    # Map model_week to year for x-axis ticks
    year_ticks = data_with_coord.groupby('year')['model_week'].first().values
    year_labels = data_with_coord['year'].unique()

    # Update x-axis ticks for the plot
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    return fig

def plot_step_based_approaches_by_year(yearly_data_with_coord, yearly_data_without_coord):
    """
    Plot step-based approaches by year as two separate stacked bar charts: one for data with a coordinator and one without.

    Args:
        yearly_data_with_coord (pd.DataFrame): Aggregated data by year with step-based approaches (with coordinator).
        yearly_data_without_coord (pd.DataFrame): Aggregated data by year with step-based approaches (without coordinator).

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
        list of matplotlib.axes.Axes: The axes objects for the subplots.
    """
    # Defensive copy and preparation: sort by year, fill NaNs, ensure numeric
    yw = yearly_data_with_coord.copy()
    yw = yw.sort_values('year').reset_index(drop=True)
    yw[['step_micros_approached_randomly',
        'step_micros_approached_recommended',
        'step_micros_approached_coordinator']] = (
        yw[['step_micros_approached_randomly',
            'step_micros_approached_recommended',
            'step_micros_approached_coordinator']]
        .fillna(0)
        .apply(pd.to_numeric)
    )

    yn = yearly_data_without_coord.copy()
    yn = yn.sort_values('year').reset_index(drop=True)
    yn[['step_micros_approached_randomly',
        'step_micros_approached_recommended']] = (
        yn[['step_micros_approached_randomly',
            'step_micros_approached_recommended']]
        .fillna(0)
        .apply(pd.to_numeric)
    )

    # Create one figure with two axes and use numeric x positions to avoid categorical misalignment
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for data with coordinator
    ax = axes[0]
    xw = np.arange(len(yw))
    width = 0.8
    ax.bar(xw, yw['step_micros_approached_randomly'], width, label='Randomly Approached')
    ax.bar(xw, yw['step_micros_approached_recommended'], width,
           bottom=yw['step_micros_approached_randomly'], label='Recommended')
    ax.bar(xw, yw['step_micros_approached_coordinator'], width,
           bottom=(yw['step_micros_approached_randomly'] + yw['step_micros_approached_recommended']),
           label='Approached via Coordinator')
    ax.set_title('With Coordinator')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Approaches')
    ax.set_xticks(xw)
    ax.set_xticklabels(yw['year'])
    ax.legend(loc='upper left')

    # Plot for data without coordinator
    ax = axes[1]
    xn = np.arange(len(yn))
    ax.bar(xn, yn['step_micros_approached_randomly'], width, label='Randomly Approached')
    ax.bar(xn, yn['step_micros_approached_recommended'], width,
           bottom=yn['step_micros_approached_randomly'], label='Recommended')
    ax.set_title('Without Coordinator')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Approaches')
    ax.set_xticks(xn)
    ax.set_xticklabels(yn['year'])
    ax.legend(loc='upper left')

    return fig

# params = {"over_65_population": 1412, "num_years": 15, "p_promote_micro": 0.1, "p_review_care": 0.1}
# results = run_care_model(params)

# params = {"over_65_population": 1412, "num_years": 15, "n_coordinators": 1}
# results2 = run_care_model(params)

# coord_data, non_coord_data = prepare_visualization_data(results2, results)

# # plot_percent_receiving_care(coord_data, non_coord_data)

# coord_yearly_data = aggregate_step_approaches_by_year(coord_data)
# non_coord_yearly_data = aggregate_step_approaches_by_year(non_coord_data)
# # plot_step_based_approaches_by_year(coord_yearly_data, non_coord_yearly_data)

# # plot_quality_over_time(coord_data, non_coord_data)

# plot_step_based_approaches_by_year(coord_yearly_data, non_coord_yearly_data)

# plt.show()