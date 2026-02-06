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
    Extracts relationships between agents, including their positions.

    Parameters:
    -----------
    coord_data : dict or None
        Coordinator data dictionary. Can be None if no coordinator exists.
    micro_data : dict
        Microprovider data dictionary.
    res_data : dict
        Resident data dictionary.

    Returns:
    --------
    relationships : list
        List of relationship dictionaries.
    positions : dict
        Dictionary of agent positions.
    """
    relationships = []
    positions = {}

    # Add coordinator relationships and positions if coordinator exists
    if isinstance(coord_data, pd.DataFrame):
        if not coord_data.empty:
            for _, row in coord_data.iterrows():
                coord_id = row['agent_id']
                pos = row.get('pos', None)
                if isinstance(pos, (tuple, list)) and len(pos) == 2 and all(isinstance(coord, (int, float)) for coord in pos):
                    positions[f'C{coord_id}'] = tuple(pos)  # Valid position
                else:
                    positions[f'C{coord_id}'] = (0, 0)  # Default position
                managed_microproviders = row['registered_microproviders']
                if isinstance(managed_microproviders, list):
                    for mp_id in managed_microproviders:
                        relationships.append({
                            'entity1': f'C{coord_id}',
                            'entity2': f'MP{mp_id}',
                            'relationship_type': 'manages/registers',
                            'source': 'coordinator_register'
                        })

    # Add microprovider relationships and positions
    micro_df = micro_data if isinstance(micro_data, pd.DataFrame) else pd.DataFrame(micro_data)
    for _, row in micro_df.iterrows():
        mp_id = row['agent_id']
        pos = row.get('pos', None)
        if isinstance(pos, (tuple, list)) and len(pos) == 2 and all(isinstance(coord, (int, float)) for coord in pos):
            positions[f'MP{mp_id}'] = tuple(pos)  # Valid position
        else:
            positions[f'MP{mp_id}'] = (0, 0)  # Default position
        allocated_residents = row['allocated_residents']
        if isinstance(allocated_residents, list) and allocated_residents:
            for res_id in allocated_residents:
                relationships.append({
                    'entity1': f'MP{mp_id}',
                    'entity2': f'R{res_id}',
                    'relationship_type': 'provides_care',
                    'source': 'microprovider_register'
                })

    # Add resident relationships and positions
    res_df = res_data if isinstance(res_data, pd.DataFrame) else pd.DataFrame(res_data)
    for _, row in res_df.iterrows():
        res_id = row['agent_id']
        pos = row.get('pos', None)
        if isinstance(pos, (tuple, list)) and len(pos) == 2 and all(isinstance(coord, (int, float)) for coord in pos):
            positions[f'R{res_id}'] = tuple(pos)  # Valid position
        else:
            positions[f'R{res_id}'] = (0, 0)  # Default position
        allocated_mps = row['allocated_microproviders']
        if isinstance(allocated_mps, list) and allocated_mps:
            for mp_id in allocated_mps:
                relationships.append({
                    'entity1': f'R{res_id}',
                    'entity2': f'MP{mp_id}',
                    'relationship_type': 'receives_care_from',
                    'source': 'resident_register'
                })

    return relationships, positions

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

    Parameters:
    -----------
    coordinator_data, microprovider_data, resident_data : dict
        Data dictionaries for coordinators, microproviders, and residents.

    Returns:
    --------
    G : networkx.MultiGraph
        The constructed network graph.
    pos : dict
        Node position dictionary for visualization.
    coordinators, microproviders, residents : list
        Lists of nodes by type for coloring.
    """
    relationships, positions = extract_relationships(coordinator_data, microprovider_data, resident_data)
    G = nx.MultiGraph()

    # Add nodes with types
    nodes_added = set()
    coordinators = []
    microproviders = []
    residents = []
    for rel in relationships:
        for entity in rel['entity1'], rel['entity2']:
            if entity not in nodes_added:
                if entity.startswith('C'):
                    G.add_node(
                        entity,
                        type='coordinator',
                        label=f"Coordinator {entity[1:]}")
                    coordinators.append(entity)
                elif entity.startswith('MP'):
                    G.add_node(
                        entity,
                        type='microprovider',
                        label=f"Microprovider {entity[2:]}")
                    microproviders.append(entity)
                elif entity.startswith('R'):
                    G.add_node(
                        entity,
                        type='resident',
                        label=f"Resident {entity[1:]}")
                    residents.append(entity)
                nodes_added.add(entity)

    # Add edges
    for rel in relationships:
        G.add_edge(rel['entity1'], rel['entity2'],
                   relationship=rel['relationship_type'],
                   source=rel['source'])

    return G, positions, coordinators, microproviders, residents

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

    # Validate positions
    for node, position in pos.items():
        if not (isinstance(position, tuple) and len(position) == 2 and all(isinstance(coord, (int, float)) for coord in position)):
            raise ValueError(f"Invalid position for node {node}: {position}")

    fig, ax = plt.subplots(figsize=(16, 12))

    # Highlight microproviders based on their degree (number of connections)
    microprovider_degrees = {node: G.degree(node) for node in microproviders}
    microprovider_sizes = [1200 + 50 * microprovider_degrees[node] for node in microproviders]  # Scale size by degree
    microprovider_colors = [0.5 + 0.5 * (microprovider_degrees[node] / max(microprovider_degrees.values())) for node in microproviders]  # Scale color intensity

    # Adjust edge opacity based on the degree of connected nodes
    edge_alphas = []
    for edge in G.edges():
        node1, node2 = edge[:2]
        degree1 = G.degree(node1)
        degree2 = G.degree(node2)
        avg_degree = (degree1 + degree2) / 2
        max_degree = max(dict(G.degree()).values())
        edge_alphas.append(0.2 + 0.8 * (avg_degree / max_degree))  # Scale alpha between 0.2 and 1.0

    # Highlight edges between coordinators and microproviders
    highlighted_edges = [
        (u, v) for u, v in G.edges()
        if (u in coordinators and v in microproviders) or (v in coordinators and u in microproviders)
    ]

    # Highlight edges between microproviders
    microprovider_edges = [
        (u, v) for u, v in G.edges()
        if u in microproviders and v in microproviders
    ]

    # Highlight edges between residents and microproviders
    resident_microprovider_edges = [
        (u, v) for u, v in G.edges()
        if (u in residents and v in microproviders) or (v in residents and u in microproviders)
    ]

    # Draw resident-to-microprovider edges with low opacity
    nx.draw_networkx_edges(
        G, pos, edgelist=resident_microprovider_edges, edge_color='gray', alpha=0.2, ax=ax, label='Resident-Microprovider'
    )

    # Draw microprovider-to-microprovider edges in amber
    nx.draw_networkx_edges(
        G, pos, edgelist=microprovider_edges, edge_color='orange', width=2.5, ax=ax, label='Microprovider-Microprovider'
    )

    # Draw coordinator-to-microprovider edges in red
    nx.draw_networkx_edges(
        G, pos, edgelist=highlighted_edges, edge_color='red', width=2.5, ax=ax, label='Coordinator-Microprovider'
    )

    # Draw other edges with varying opacity
    other_edges = [
        (u, v) for u, v in G.edges() if (u, v) not in highlighted_edges and (u, v) not in microprovider_edges and (u, v) not in resident_microprovider_edges
    ]
    edges = nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='black', alpha=edge_alphas, ax=ax)

    # Set edge transparency for other edges
    for edge, alpha in zip(edges, edge_alphas):
        edge.set_alpha(alpha)

    # Draw nodes by type
    nx.draw_networkx_nodes(G, pos, nodelist=microproviders, node_color=microprovider_colors,
                           cmap=plt.cm.Blues, node_size=microprovider_sizes, node_shape='o', label='Microproviders', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=residents, node_color='lightgreen',
                           node_size=800, node_shape='^', label='Residents', ax=ax)

    # Draw coordinator nodes last to ensure they are on top
    nx.draw_networkx_nodes(G, pos, nodelist=coordinators, node_color='lightcoral',
                           node_size=2000, node_shape='s', label='Coordinators', ax=ax)

    # Add labels only for coordinators
    labels = {node: node for node in coordinators}  # Only label coordinators
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

    # Add title and legend
    ax.set_title('Care System Network', fontsize=16, fontweight='bold', pad=30)
    # Adjust legend to be between the title and the network figure
    legend = ax.legend(scatterpoints=1, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    legend.set_title('Legend', prop={'size': 14})

    # Reduce the size of markers in the legend
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([40])  # Reduce marker size
        elif hasattr(handle, 'set_width') and hasattr(handle, 'set_height'):
            handle.set_width(0.5)  # Reduce polygon width
            handle.set_height(0.5)  # Reduce polygon height

    # Add a horizontal color bar to indicate the scale of connectivity for microproviders
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(microprovider_degrees.values()), vmax=max(microprovider_degrees.values())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Microprovider Connectivity', fontsize=16, fontweight='bold')

    # Remove the border from the figure
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig