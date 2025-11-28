import mesa
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import requests
import os
import pickle
import pprint
import logging
from io import StringIO
from datetime import datetime
import networkx as nx
import ast
from collections import defaultdict
from mesa import Agent, Model
from mesa.time import RandomActivation # random order of agent actions
from mesa.space import MultiGrid # multiple agents er cell
from mesa.datacollection import DataCollector

# from model import run_care_model

def extract_relationships(coord_data, micro_data, res_data):
    """
    Extracts relationships between agents, handling cases with no coordinator.
    
    Parameters:
    -----------
    coord_data : dict or None
        Coordinator data dictionary. Can be None if no coordinator exists.
    micro_data : dict
        Microprovider data dictionary
    res_data : dict
        Resident data dictionary
    
    Returns:
    --------
    relationships : list
        List of relationship dictionaries
    """
    relationships = []
    
    # Add coordinator relationships if coordinator exists
    if isinstance(coord_data, pd.DataFrame):
        if not coord_data.empty:
            coordinator_id = coord_data['agent_id'].iloc[0]
            managed_microproviders = coord_data['registered_microproviders'].iloc[0]
            if isinstance(managed_microproviders, list):
                for mp_id in managed_microproviders:
                    relationships.append({
                        'entity1': f'C{coordinator_id}',
                        'entity2': f'MP{mp_id}',
                        'relationship_type': 'manages/registers',
                        'source': 'coordinator_register'
                    })
    
    # Convert to DataFrame only if needed
    micro_df = micro_data if isinstance(micro_data, pd.DataFrame) else pd.DataFrame(micro_data)
    for _, row in micro_df.iterrows():
        mp_id = row['agent_id']
        allocated_residents = row['allocated_residents']
        if isinstance(allocated_residents, list) and allocated_residents:  # Check if list and not empty
            for res_id in allocated_residents:
                relationships.append({
                    'entity1': f'MP{mp_id}',
                    'entity2': f'R{res_id}',
                    'relationship_type': 'provides_care',
                    'source': 'microprovider_register'
                })
    
    # Convert to DataFrame only if needed            
    res_df = res_data if isinstance(res_data, pd.DataFrame) else pd.DataFrame(res_data)
    for _, row in res_df.iterrows():
        res_id = row['agent_id']
        allocated_mps = row['allocated_microproviders']
        if isinstance(allocated_mps, list) and allocated_mps:  # Check if list and not empty
            for mp_id in allocated_mps:
                relationships.append({
                    'entity1': f'R{res_id}',
                    'entity2': f'MP{mp_id}',
                    'relationship_type': 'receives_care_from',
                    'source': 'resident_register'
                })
                
    return relationships

def create_layout(G, coordinators, microproviders, residents):
    """
    Creates a custom layout that works with or without coordinators.
    """
    pos = {}
    
    # Handle coordinator positioning if they exist
    if coordinators:
        for coord in coordinators:
            pos[coord] = (0, 0)
        
        # Separate microproviders into connected and unconnected
        connected_mps = []
        unconnected_mps = []
        for mp in microproviders:
            if any(coord in G.neighbors(mp) for coord in coordinators):
                connected_mps.append(mp)
            else:
                unconnected_mps.append(mp)
                
        # Position connected microproviders
        mp_radius = 2
        for i, mp in enumerate(connected_mps):
            angle = 2 * np.pi * i / (len(connected_mps) or 1)
            pos[mp] = (mp_radius * np.cos(angle), mp_radius * np.sin(angle))
            
        # Position unconnected microproviders
        unconnected_radius = 4
        for i, mp in enumerate(unconnected_mps):
            angle = np.pi/3 + (np.pi/3 * i / (len(unconnected_mps) or 1))
            pos[mp] = (unconnected_radius * np.cos(angle), unconnected_radius * np.sin(angle))
            
        # Position residents
        res_radius = 4
        for i, res in enumerate(residents):
            mp_neighbors = [n for n in G.neighbors(res) if n.startswith('MP')]
            if mp_neighbors:
                mp_pos = pos[mp_neighbors[0]]
                angle = 2 * np.pi * i / len(residents)
                offset = 1.5
                pos[res] = (mp_pos[0] + offset * np.cos(angle),
                           mp_pos[1] + offset * np.sin(angle))
            else:
                angle = 2 * np.pi * i / len(residents)
                pos[res] = (res_radius * np.cos(angle), res_radius * np.sin(angle))
    else:
        # Alternative layout for no coordinator scenario
        # Place microproviders in a circle
        mp_radius = 3
        for i, mp in enumerate(microproviders):
            angle = 2 * np.pi * i / (len(microproviders) or 1)
            pos[mp] = (mp_radius * np.cos(angle), mp_radius * np.sin(angle))
        
        # Place residents near their connected microproviders
        res_radius = 5
        for i, res in enumerate(residents):
            mp_neighbors = [n for n in G.neighbors(res) if n.startswith('MP')]
            if mp_neighbors:
                mp_pos = pos[mp_neighbors[0]]
                angle = 2 * np.pi * i / len(residents)
                offset = 1.5
                pos[res] = (mp_pos[0] + offset * np.cos(angle),
                           mp_pos[1] + offset * np.sin(angle))
            else:
                angle = 2 * np.pi * i / len(residents)
                pos[res] = (res_radius * np.cos(angle), res_radius * np.sin(angle))
    
    return pos

def create_network_graph(coordinator_data, microprovider_data, resident_data):
    """
    Creates a network graph visualization of the care system relationships.
    
    This function takes three data dictionaries representing different agents in the care system
    and creates a NetworkX MultiGraph showing their relationships. The visualization places:
    - Coordinator in the center (square node)
    - Connected microproviders in an inner circle (circular nodes)
    - Unconnected microproviders in a separate cluster (circular nodes)
    - Residents near their connected microproviders (triangle nodes)
    
    Parameters:
    -----------
    coordinator_data : dict
        Dictionary containing coordinator information including:
        - agent_id: list of coordinator IDs
        - registered_microproviders: list of lists containing managed microprovider IDs
        - micro_quality_threshold: quality thresholds for microproviders
    
    microprovider_data : dict
        Dictionary containing microprovider information including:
        - agent_id: list of microprovider IDs
        - allocated_residents: list of lists containing resident IDs they serve
        - other metadata (capacity, quality, etc.)
    
    resident_data : dict
        Dictionary containing resident information including:
        - agent_id: list of resident IDs
        - allocated_microproviders: list of lists containing assigned microprovider IDs
        - other metadata (care needs, packages received, etc.)
    
    Returns:
    --------
    G : networkx.MultiGraph
        The constructed network graph
    pos : dict
        Node position dictionary for visualization
    """
    relationships = extract_relationships(coordinator_data, microprovider_data, resident_data)
    G = nx.MultiGraph()
    
    # Add nodes with types
    nodes_added = set()
    for rel in relationships:
        for entity in [rel['entity1'], rel['entity2']]:
            if entity not in nodes_added:
                if entity.startswith('C'):
                    G.add_node(
                        entity,
                        type='coordinator',
                        label=f"Coordinator {entity[1:]}")
                elif entity.startswith('MP'):
                    G.add_node(
                        entity,
                        type='microprovider',
                        label=f"Microprovider {entity[2:]}")
                elif entity.startswith('R'):
                    G.add_node(
                        entity,
                        type='resident',
                        label=f"Resident {entity[1:]}")
                nodes_added.add(entity)
                
    # Add edges
    for rel in relationships:
        G.add_edge(rel['entity1'], rel['entity2'],
                  relationship=rel['relationship_type'],
                  source=rel['source'])
                  
    # Get node lists by type
    coordinators = [n for n, d in G.nodes(data=True) if d['type'] == 'coordinator']
    microproviders = [n for n, d in G.nodes(data=True) if d['type'] == 'microprovider']
    residents = [n for n, d in G.nodes(data=True) if d['type'] == 'resident']
    
    # Create layout
    pos = create_layout(G, coordinators, microproviders, residents)
    
    return G, pos, coordinators, microproviders, residents

def visualize_network(G, pos, coordinators, microproviders, residents):
    """
    Creates a visualization of the network graph.
    
    Parameters:
    -----------
    G : networkx.MultiGraph
        The network graph to visualize
    pos : dict
        Dictionary of node positions
    coordinators, microproviders, residents : list
        Lists of nodes by type for coloring
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the network visualization
    """
    print("Visualize network function called")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='black', alpha=1, ax=ax)
    
    # Draw nodes by type
    nx.draw_networkx_nodes(G, pos, nodelist=coordinators, node_color='lightcoral',
                           node_size=1500, node_shape='s', label='Coordinators', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=microproviders, node_color='lightblue',
                           node_size=1200, node_shape='o', label='Microproviders', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=residents, node_color='lightgreen',
                           node_size=800, node_shape='^', label='Residents', ax=ax)
    
    # Add labels
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Add title and legend
    ax.set_title('Care System Network', fontsize=16, fontweight='bold')
    ax.legend(scatterpoints=1)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

'''calling functions'''

# from model import run_care_model

# # Define parameters for the model
# params = {
#     "n_residents": 300,  # Example number of residents
#     "n_microproviders": 10,  # Example number of microproviders
#     "n_coordinators": 1,  # Example number of coordinators
#     "annual_population_growth_rate": 0.011,  # 1.1% annual growth
#     "p_resident_leave": 0.01,  # Probability of residents leaving
#     "p_microprovider_leave": 0.01,  # Probability of microproviders leaving
#     "num_years": 10,  # Number of years to simulate
# }

# # Run the care model
# results = run_care_model(params)

# # Extract data from the model results
# coordinator_data = results.get("data_coord_registry", pd.DataFrame())
# microprovider_data = results.get("data_microprovider_registry", pd.DataFrame())
# resident_data = results.get("data_resident_registry", pd.DataFrame())

# # Create the network graph
# G, pos, coordinators, microproviders, residents = create_network_graph(
#     coordinator_data, microprovider_data, resident_data
# )

# # Visualize the network
# fig = visualize_network(G, pos, coordinators, microproviders, residents)
# plt.show()