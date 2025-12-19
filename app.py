import mesa
import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import pprint
import logging
from io import StringIO
from datetime import datetime
import networkx as nx
import ast
from collections import defaultdict
from PIL import Image
from mesa import Agent, Model
from mesa.time import RandomActivation  # random order of agent actions
from mesa.space import MultiGrid  # multiple agents per cell
from mesa.datacollection import DataCollector
from model import run_care_model
from network_analysis import create_network_graph, visualize_network
from analysis import prepare_visualization_data, plot

# Title of the app
st.set_page_config(layout="wide")
st.title(f"Micro-Providers Agent Based Simulation Sandbox")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Welcome",
                        "Introduction",
                        "About the Model",
                        "Run the Model",
                        "Model Results",
                        "Network Analysis",
                        "So, What Do You Think?"]
                    )
# Home Page
if page == "Welcome":
    st.header("Header")
    st.write("Welcome to the Micro-Providers Agent Based Simulation Streamlit app./" \
    " You can navigate to different pages within the app using the sidebar.")

elif page == "Introduction":
    st.header("Find and Answer Questions")

    introducion_body = """
    
    ## What are Micro-Providers?

    A micro-provider is a small, community-based business or individual that offers flexible and personalized care or support services.

    ## What do you mean by an Agent Based Simulation?

    For our purposes, an agent based simulation describes how we are using
    computer software to create things, how we tell those things to behave,
    and then how we observe those things behaving when we run the simulation.
    Those "things" are our agents.
    
    For this simulation, the things we are creating are the residents
    who need care in a community and the micro-providers who are looking to
    care for them.

    Each time we run the simulation, we effectively run it twice: once with
    a co-ordinator agent, who helps connect micro-providers to residents who
    need care, and once more without a co-ordinator agent, to see how residents
    and micro-providers might get on themselves. A more detailed definition
    of agent based simulations and how the agents behave in the simulation can
    be found on the "about the model" page.

    ## Why bother with an Agent Based Simulation for Micro-Providers?

    Agent Based Simulations give us an opportunity to use statistics
    and behaviours we might understand to create a representation of an event
    or thing that we don't fully understand—think: social phenomena.
    Our ability to shape how agents behave in the simulation, and then see how
    social phenomena changes within the simulation, could improve our
    understanding of the phenomena in the real world too—if the simulation is
    any good!

    Our phenomena is that micro-providers are looking after people in the
    community. We have some understanding on how residents and micro-providers
    come together, and some statistics on how many older-aged adults are having
    their care needs met at home, or have unmet need. We don't fully understand
    how many micro-providers might be working within communities at any one
    time, and what the networks that form between micro-providers and residents
    look like.

    A well-designed agent based simulation, then, might give us insight into the
    number of micro-providers working in communities, how they find employment,
    and what the relationships between micro-providers and residents look like.
    This would give us a foundation to encourage people to:

    - take up micro-providing,
    - decision-makers to invest in outcomes that could increase amicro-providers access to work, and,
    - encourage communities to organise their infrastructure in a way that encourages micro-provider activity.

    ## What do you mean by 'Sandbox'?

    Simply that, this is a platform to explore an idea and to encourage us to
    think differently about how micro-providers exist in communities, and how
    we might explore agent based simulations in other areas of health and
    social care.
    """

    st.markdown(introducion_body)

elif page == "About the Model":
    st.header("About the Model")
    st.write("This section describes the agent-based model used in the simulation.")
    # interactive df with model parameters and descriptions
    tab1, tab2, tab3 = st.tabs(
        ["About the Model and Model Parameters",
         "Agent Step Behaviours",
         "Sources and Statistics"]
    )
    with tab1:
        st.subheader("About the Model and Model Parameters")
        st.write("Below is an interactive table of the model parameters and their descriptions:")

        # Define parameters and their descriptions
        params_with_explanations = {
            "civil_parish": "The selected civil parish, which determines the resident population.",
            "num_years": "Number of years to simulate.",
            "annual_population_growth_rate": "Annual population growth rate (as a percentage).",
            "p_resident_leave": "Probability of a resident leaving the model per step.",
            "p_microprovider_join": "Probability of a microprovider joining the model per step.",
            "p_microprovider_leave": "Probability of a microprovider leaving the model per step.",
            "p_use_coordinator": "Probability of a resident using a coordinator per step.",
            "p_approach_random_micro": "Probability of a resident randomly approaching a microprovider per step.",
            "p_review_care": "Probability of a resident reviewing their care per step.",
            "p_promote_micro": "Probability of a resident promoting a microprovider per step.",
            "micro_quality_threshold": "Minimum quality threshold for microproviders to join a coordinator.",
            "micro_join_coord": "Probability of a microprovider joining a coordinator per step.",
        }

        # Convert the dictionary to a DataFrame
        params_df = pd.DataFrame.from_dict(
            params_with_explanations, orient="index", columns=["Description"]
        )

        # Display the DataFrame in Tab 1
        st.subheader("Parameters Table")
        st.dataframe(params_df)

    with tab2:
        st.subheader("Agent Step Behaviours")
        st.write("Details about the agents in the model and their behaviors during each step.")

        sub_tab = st.radio("Select a Sub-Tab",
                        ["Model Step",
                        "Resident Step",
                        "Micro-Provider Step",
                        "Coordinator Step"]
                        )

        if sub_tab == "Model Step":
            model_image_url = "https://raw.githubusercontent.com/houjamr/HSMA_ABS/refs/heads/main/Model_Step.drawio.png"
            st.image(model_image_url, caption="Model Step Diagram")
        elif sub_tab == "Resident Step":
            resident_image_url = "https://raw.githubusercontent.com/houjamr/HSMA_ABS/refs/heads/main/Resident_Step.drawio.png"
            st.image(resident_image_url, caption="Resident Agent Step Diagram")
        elif sub_tab == "Micro-Provider Step":
            microprovider_image_url = "https://raw.githubusercontent.com/houjamr/HSMA_ABS/refs/heads/main/Micro-Provider_Step.drawio.png"
            st.image(microprovider_image_url, caption="Micro-Provider Agent Step Diagram")
        elif sub_tab == "Coordinator Step":
            coordinator_image_url = "https://raw.githubusercontent.com/houjamr/HSMA_ABS/refs/heads/main/Coordinator_Step.drawio.png"
            st.image(coordinator_image_url, caption="Coordinator Agent Step Diagram")

    with tab3:
        st.subheader("Sources and Statistics")
        st.write("Sources of data and statistics used in the model.")

        sources_stats_body = """
        ## Civil Parish Population Size

        **Office for National Statistics (2021)** PP012 - Age  
        Available at: https://www.nomisweb.co.uk/datasets/c2021pp012
        
        The 2021 Census provides data on the size and age of populations living
        in the United Kingdom. PP012 takes this data and applies it to
        Civil Parishes in England.
        
        The model uses this data for users to select a civil parish to
        simulate micro-provider interactions.

        ## Population Growth

        **Office for National Statistics (2022)** 
        Subnational population projections for England: 2022-based.  
        Available at: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/bulletins/subnationalpopulationprojectionsforengland/2022based
        
        The ONS provides data for population projections up to 2045.

        The model uses this data to give a representation of population growth
        throughout the model.

        ## Older Aged Adults Needing Care at Home

        **NHS Digital (2022)** Health Survey for England 2021: Social care.  
        Available at: https://digital.nhs.uk/data-and-information/publications/statistical/health-survey-for-england/2021-part-2/social-care
        
        The Health Survey for England is a longitudinal survey used to monitor
        trends in health and care for adults aged 16 or over. A constituent part
        of the survey includes data collection on older aged-adults (65+), who
        live at home, how their care needs are being met. The survey asked
        participants to self-report on their ability to perform activities of
        daily living (ADL) or instrumental activities of daily living (IADL),
        if they required help to complete these activities, or were unable to
        complete them at all. From the data, the survey produces percentage
        estimates for the number of adults by age group who have either received
        help to complete activities recently, needed help to complete I/ADLS
        or have unmet need–who need help but are not currently receiving any
        support.
        
        The model uses a proportion of the "received help" figure to serve as
        a baseline estimate of how many older-aged adults in a parish (i.e.,
        older aged adults) are receiving care from a micro-provider to warm up
        the simulation. There is no statistical evidence on what proportion of
        "received help" is delivered from micro-providers however, Somerset
        Council estimate this to be 50% of support delivered at home originates
        from micro-providers. This figure is configurable as an advanced
        parameter within the simulation.
        
        The model also uses the unmet need parameter as a limit for how much
        care is available for micro-providers in the community on the pretense
        that micro-providers wouldn't be providing more support to older-aged
        adults than the projected number of those with care and support needs.
        """
        st.markdown(sources_stats_body)
        
elif page == "Run the Model":
    st.header("Configure Simulation Parameters")

    # Create tabs for Basic and Advanced parameters
    tab1, tab2 = st.tabs(["Basic Parameters", "Advanced Parameters"])

    # Basic Parameters
    with tab1:

        # Load the CSV file
        csv_path = "https://raw.githubusercontent.com/houjamr/HSMA_ABS/refs/heads/main/SouthSomParish_65_Figs.csv"
        df = pd.read_csv(csv_path, skiprows=6)  # Skip metadata rows at the top
        df.columns = df.columns.str.strip().str.replace('"', '')
        df = df.drop(index=1)
        df = df.dropna(subset=["2022 parish"])  # Drop rows with NaN in '2022 parish'
        df = df[df["2022 parish"] != "-"]

        # Add a dropdown for parish selection
        st.subheader("Select a Parish")
        selected_parish = st.selectbox("Select a Parish", df["2022 parish"].unique())

        # Extract the population for the selected parish
        selected_population = int(df[df["2022 parish"] == selected_parish]["Aged 65 years and over"].values[0])

        # Display the selected parish and population
        st.write(f"Selected Parish: {selected_parish}")
        st.write(f"Population Size: {int(selected_population)}")

        # Add a warning if the population size is large
        LARGE_POPULATION_THRESHOLD = 1500  # Define the threshold for a large population
        if selected_population > LARGE_POPULATION_THRESHOLD:
            st.warning(
                f"The selected parish has a population of {int(selected_population)}, "
                "This will take a long time to load. Consider selecting a parish "
                "with a population less than 1500."
            )
        MEDIUM_POPULATION_THRESHOLD = 800  # Define the threshold for a medium population
        if selected_population > MEDIUM_POPULATION_THRESHOLD and selected_population <= LARGE_POPULATION_THRESHOLD:
            st.info(
                f"The selected parish has a population of {int(selected_population)}, "
                "depending on the number of years you are simulating, this may "
                "take a while to load."
            )
        # Define parameters
        params = {
            "n_residents": int(selected_population),  # Use the selected population
            "num_years": st.slider("Number of Years to Simulate", 1, 15, 10),
        }

    # Advanced Parameters
    with tab2:
        st.subheader("Advanced Parameters")
        advanced_params = {
            "n_residents": int(selected_population),  # Use the selected population

            "annual_population_growth_rate": st.slider(
                "Annual Population Growth Rate (%)", 0.0, 5.0, 1.1, 0.1
            ) / 100,

            "p_resident_leave": st.slider(
                "Probability Resident Leaves the Model (Per Step)", 0.0, 1.0, 0.01, 0.01
            ),

            "p_microprovider_join": st.slider(
                "Probability Microprovider Joins (Per Step)", 0.0, 1.0, 0.01, 0.01
            ),

            "p_microprovider_leave": st.slider(
                "Probability Microprovider Leaves (Per Step)", 0.0, 1.0, 0.01, 0.01
            ),
            "p_use_coordinator": st.slider(
                "Probability a Resident Uses a Coordinator (Per Step)", 0.0, 1.0, 0.02, 0.01
            ),
            "p_approach_random_micro": st.slider(
                "Probability a Resident Randomly Approaches a Microprovider (Per Step)",
                0.0,
                1.0,
                0.01,
                0.01,
            ),
            "p_review_care": st.slider(
                "Probability Review Care (Per Step)", 0.0, 1.0, 0.2, 0.01
            ),
            "p_promote_micro": st.slider(
                "Probability Promote Microprovider (Per Step)", 0.0, 1.0, 0.2, 0.01
            ),
            "micro_quality_threshold": st.slider(
                "Microprovider Quality Threshold to Join a Coordinator (Per Step)", 0.0, 1.0, 0.5, 0.01
            ),
            "micro_join_coord": st.slider(
                "Probability Microprovider Joins Coordinator (Per Step)",
                0.0,
                1.0,
                0.5,
                0.01,
            ),
        }

    # Combine basic and advanced parameters into a single dictionary
    params.update(advanced_params)

    # Add a button to run the simulation with and without a coordinator
    if st.button("Run Simulation"):
        with st.spinner("Running the care model simulation..."):
            # Run the simulation with a coordinator
            params["n_coordinators"] = 1  # Set coordinators to 1
            st.write("Running simulation with coordinator...")
            results_with_coordinator = run_care_model(params)
            st.write("Simulation with coordinator complete. Preparing visualization data...")
            data_with_coordinator = prepare_visualization_data(results_with_coordinator)
            st.write("Visualization data prepared, running simulation without coordinator...")

            # Run the simulation without a coordinator
            params["n_coordinators"] = 0  # Set coordinators to 0
            results_without_coordinator = run_care_model(params)
            st.write("Simulation without coordinator complete. Preparing visualization data...")
            data_without_coordinator = prepare_visualization_data(results_without_coordinator)

            # Store results and data in session state
            st.session_state["results_with_coordinator"] = results_with_coordinator
            st.session_state["data_with_coordinator"] = data_with_coordinator
            st.session_state["results_without_coordinator"] = results_without_coordinator
            st.session_state["data_without_coordinator"] = data_without_coordinator

        st.success("Simulation complete!")

elif page == "Model Results":
    st.header("Simulation Results")
    if "results_with_coordinator" in st.session_state and "results_without_coordinator" in st.session_state:
        col1, col2 = st.columns(2)

        # Results with Coordinator
        with col1:
            st.title("With Coordinator")
            data_with_coordinator = st.session_state["data_with_coordinator"]
            fig_with_coordinator = plot(data_with_coordinator)
            st.pyplot(fig_with_coordinator)

        # Results without Coordinator
        with col2:
            st.title("Without Coordinator")
            data_without_coordinator = st.session_state["data_without_coordinator"]
            fig_without_coordinator = plot(data_without_coordinator)
            st.pyplot(fig_without_coordinator)
    else:
        st.write("No simulation results found. Please run the simulation first.")

elif page == "Network Analysis":
    st.header("Network Analysis")

    # Check if the simulation has been run
    if "results_with_coordinator" in st.session_state and "results_without_coordinator" in st.session_state:
        col1, col2 = st.columns(2)

        # Network Analysis with Coordinator
        with col1:
            st.subheader("Network Analysis (With Coordinator)")
            results_with_coordinator = st.session_state["results_with_coordinator"]
            G, pos, coordinators, microproviders, residents = create_network_graph(
                results_with_coordinator["data_coord_registry"],
                results_with_coordinator["data_microprovider_registry"],
                results_with_coordinator["data_resident_registry"]
            )
            fig = visualize_network(G, pos, coordinators, microproviders, residents)
            st.pyplot(fig)

        # Network Analysis without Coordinator
        with col2:
            st.subheader("Network Analysis (Without Coordinator)")
            results_without_coordinator = st.session_state["results_without_coordinator"]
            G, pos, coordinators, microproviders, residents = create_network_graph(
                results_without_coordinator["data_coord_registry"],
                results_without_coordinator["data_microprovider_registry"],
                results_without_coordinator["data_resident_registry"]
            )
            fig = visualize_network(G, pos, coordinators, microproviders, residents)
            st.pyplot(fig)
    else:
        # Display a message if the simulation hasn't been run
        st.write("No simulation results found. Please run the simulation first.")

elif page == "So, What Do You Think?":
    st.header("Feedback")
    st.write("Coming Soon!")