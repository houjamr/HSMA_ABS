
import glob
import mesa
import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import logging
from io import StringIO
from datetime import datetime
import networkx as nx
import json
import ast
from collections import defaultdict
from PIL import Image
from mesa import Agent, Model
from mesa.time import RandomActivation  # random order of agent actions
from mesa.space import MultiGrid  # multiple agents per cell
from mesa.datacollection import DataCollector
from model import run_care_model
from network_analysis import create_network_graph, visualize_network

from analysis import prepare_visualization_data,\
      plot_percent_receiving_care, approaches_plot,\
      aggregate_step_approaches_by_year,\
      plot_quality_over_time, plot_quality_threshold_over_time,\
      plot_step_based_approaches_by_year

# Title of the app
st.set_page_config(layout="wide")
st.title(f"Can You Dig it? A Micro-Provider Sandbox")
st.subheader("Can Agent-Based Simulations Help Us Understand and\
              Support Micro-Providers and Their Communities?")
# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Welcome and Introduction",
                        "About the Model",
                        "Run the Model",
                        "Model Results",
                        "Network Analysis",
                        "Do You Dig it?",
                        "Thank Yous, Community Groups in Somerset, and Get in Touch!"]
                    )
# Home Page
if page == "Welcome and Introduction":
    st.info("Welcome to the 'Can You Dig it? A Micro-Provider Sandbox' Streamlit app." \
    " You can navigate to different pages within the app using the sidebar.")

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
    their care needs met at home, or need help with meeting their care needs.
    We don't fully understandvhow many micro-providers might be working within
    communities at any one time, and what the networks that form between
    micro-providers and their commmunities might look like.

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
    tab1, tab2 = st.tabs(
        [
         "Agent Step Behaviours",
         "Sources and Statistics"]
    )

    with tab1:
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

    with tab2:
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
        trends in health and care for adults aged 16 or over. One part
        of the survey includes data collection on older aged-adults (65+) who
        live at home, and how their care needs are being met. The survey asked
        participants to self-report on their ability to perform activities of
        daily living (ADL) or instrumental activities of daily living (IADL),
        if they required help to complete these activities, or were unable to
        complete them at all. From the data, the survey produces percentage
        estimates for the number of adults by age group who have either received
        help to complete activities recently, needed help to complete I/ADLS
        or have unmet need in completing I/ADLs.
        
        The model uses a proportion of the "received help" figure to serve as
        a baseline estimate of how many older-aged adults in a parish (i.e., residents)
        are receiving care from a micro-provider.
         
        There is no readily available evidence on what proportion of
        "received help" is delivered from micro-providers however, Somerset
        Council estimates that up to 50% of support delivered at home originates
        from micro-providers. Where the HSE survey returns ~10% of older-aged adults
        are receiving care at home, this would mean ~5% are receiving care from
        micro-providers in Somerset.
        This figure is configurable as an advanced parameter within the simulation.

        The model uses the "need help" figure to serve as a baseline estimate of 
        how many older-aged adults need care but are not yet receiving it.

        The total population generated then at the start of the simulation is the 
        % receiving help + % needing help, which will then gradually increase over
        time with population growth.
        """

        st.markdown(sources_stats_body)
        
elif page == "Run the Model":
    st.info("Not interested in setting parameters? We have some default data, just press run at the bottom of the page!")
    st.header("How do we run the model?")

    run_model_body = """
                When we run the model, we are running two simulations, one with
                a coordinator agent active, and one without. All other
                parameters will remain the same, as we configure them below. 
                
                
                If you do not want to configure any parameters yourself,
                there are default parameters set, so just run the model and see what happens!

                1) First we select a civil parish from the dropdown menu, this gives
                us the population size of older-aged adults (65+) in a Civil Parish.

                2) Next we configure the number of older-aged adults who are receiving care 
                at home from Micro-Providers, and the number of older-aged adults
                who still need help out in the community.

                3) Next we select what behaviours we would like our coordinator 
                to have in the simulation they are active in.

                4) Finally, we select the number of years we would like to simulate.
                Population estimates currently only run to 2040, so we can simulate
                a maximum of 15 (2025-2040) with the current population growth parameter.

                5) You can configure other parameters that
                shape how that effect the outcomes of the simulations in the
                "Advanced Parameters" tab. (Optional).
                """
    
    st.markdown(run_model_body)

    st.header("Configure Simulation Parameters")

    # Create tabs for Basic and Advanced parameters
    tab1, tab2 = st.tabs(["Basic Parameters", "Advanced Parameters"])

    # Basic Parameters
    with tab1:
        csv_files = glob.glob("*Parish_65_Figs.csv")
        if not csv_files:
            # Fallback to the remote South Somerset file if no local files found
            csv_files = ["https://raw.githubusercontent.com/houjamr/HSMA_ABS/refs/heads/main/SouthSomParish_65_Figs.csv"]

        parts = []
        for fp in csv_files:
            try:
                part = pd.read_csv(fp, skiprows=6, header=0)
            except Exception:
                # try without skiprows if reading from a remote URL that already has header
                part = pd.read_csv(fp)
            parts.append(part)

        df = pd.concat(parts, ignore_index=True, sort=False)
        df.columns = df.columns.str.strip().str.replace('"', '')
        # drop the extra metadata row if present
        df = df.dropna(subset=["2022 parish"])  # Drop rows with NaN in '2022 parish'
        df = df[df["2022 parish"] != "-"]

        # Ensure population column is numeric and fill or coerce missing
        df["Aged 65 years and over"] = pd.to_numeric(df["Aged 65 years and over"], errors="coerce").fillna(0).astype(int)

        # Drop parishes with fewer than 100 older-aged adults
        df = df[df["Aged 65 years and over"] >= 100]

        # Build display labels with population and sort by population (low to high)
        df = df.drop_duplicates(subset=["2022 parish"])  # avoid duplicates across district files
        df = df.sort_values(by="Aged 65 years and over", ascending=True)
        df["display"] = df["2022 parish"] + " (" + df["Aged 65 years and over"].astype(str) + ")"

        # Add a dropdown for parish selection (sorted by population)
        st.subheader("Configure Population size")
        st.write("Select the parameters which control the population size of the simulations.")
        display_options = df["display"].tolist()
        # Default the selection to Wincanton (if present)
        default_index = 0
        try:
            winc_row = df[df["2022 parish"].str.lower() == "wincanton"]
            if not winc_row.empty:
                default_display = winc_row.iloc[0]["display"]
                default_index = display_options.index(default_display)
        except Exception:
            default_index = 0

        selected_display = st.selectbox(
            "Select a Civil Parish - ordered by population (low to high where population is > 100)",
            display_options,
            index=default_index,
        )

        # Extract the population for the selected parish
        selected_row = df[df["display"] == selected_display].iloc[0]
        selected_parish = selected_row["2022 parish"]
        selected_population = int(selected_row["Aged 65 years and over"])

        # Display the selected parish and population
        st.write(f"Selected Parish is {selected_parish} with {int(selected_population)} older-aged adults (65+).")

        # Define parameters and set them immediately on `params`
        params = {
            "over_65_population": int(selected_population),  # Use the selected population
        }

        params["target_care_percentage"] = st.number_input(
            "Baseline percentage of residents already receiving help from micro-providers (%)", 0, 100, 5, 1
        )

        params["needs_help_percentage"] = st.number_input(
            "Baseline percentage of residents needing help but not receiving it (%)", 0, 100, 26, 1
        )

        # Compute initial placements and present them clearly
        num_receiving = int(params['over_65_population'] * params['target_care_percentage'] / 100)
        num_needing = int(params['over_65_population'] * params['needs_help_percentage'] / 100)
        st.write(
            f"Initial placements: {num_receiving} residents will start with micro-providers. "
            f"Additionally, {num_needing} residents currently need help but are not receiving it."
        )

        # Show a performance warning when warm-up will place many agents
        projected_agents = num_receiving + num_needing
        if projected_agents > 1000:
            st.warning(
                f"Warning: the model will place {projected_agents} agents during warm-up — this may be slow. "
                "Consider reducing the selected population or percentages."
            )

        st.subheader("Configure Coordinator Behaviours")
        st.write("Select the behaviours you would like the coordinator to\
                  have in the simulation they are active in.")
        
        # Coordinator behaviour options - set directly on params
        params["coord_microprovider_outreach"] = st.checkbox(
            "Coordinator does microprovider outreach", value=True)
        params["coord_run_resident_peer_support_group"] = st.checkbox(
            "Coordinator runs resident peer support group", value=True)
        params["coord_promotes_microproviders"] = st.checkbox(
            "Coordinator promotes microproviders", value=True)
        params["coord_recommends_microproviders"] = st.checkbox(
            "Coordinator recommends microproviders to residents", value=True)

        st.subheader("Simulation Parameters")
        st.write("Select the number of years you would like to simulate.")
        params["num_years"] = st.slider("Number of Years to Simulate", 1, 15, 10)

    # Advanced Parameters
    with tab2:
        st.subheader("Advanced Parameters")
        st.subheader("Model advanced parameters")
        model_advanced_params = {

            "annual_population_growth_rate": st.number_input(
                "Annual Population Growth Rate (%)", 0.0, 5.0, 1.1, 0.1
            ) / 100,

            "p_resident_leave": st.number_input(
                "Probability Resident Leaves the Model (Per Step)", 0.0, 0.1, 0.01, 0.01)
            ,

            "p_microprovider_join": st.number_input(
                "Probability Microprovider Joins (Per Step)", 0.0, 1.0, 0.01, 0.01)
            ,

            "p_microprovider_leave": st.number_input(
                "Probability Microprovider Leaves (Per Step)", 0.0, 1.0, 0.01, 0.01
            )
        }
        # immediately merge model advanced parameters
        params.update(model_advanced_params)

        st.subheader("Resident advanced parameters")
        resident_advanced_params = {
            "p_use_coordinator": st.number_input(
                "Probability a Resident Uses a Coordinator (Per Step)", 0.0, 1.0, 0.02, 0.01
            ),
            "p_approach_random_micro": st.number_input(
                "Probability a Resident Randomly Approaches a Microprovider (Per Step)",
                0.0,
                1.0,
                0.01,
                0.01,
            ),
            "p_review_care": st.number_input(
                "Probability Review Care (Per Step)", 0.0, 1.0, 0.2, 0.01
            ),
            "p_promote_micro": st.number_input(
                "Probability Promote Microprovider (Per Step)", 0.0, 1.0, 0.2, 0.01
            )
        }
        params.update(resident_advanced_params)

        st.subheader("Micro-Provider advanced parameters")
        microprovider_advanced_params = {
            "micro_join_coord": st.number_input(
                "Probability Microprovider Joins Coordinator (Per Step)",
                0.0,
                1.0,
                0.01,
            ),
        }

        params.update(microprovider_advanced_params)

        st.subheader("Coordinator advanced parameters")
        coordinator_advanced_params = {"micro_quality_threshold": st.number_input(
                "Microprovider Quality Threshold to Join a Coordinator",
                0.0, 1.0, 0.7, 0.01
            ), "resident_coordinator_group_interval": st.number_input(
                "Resident Coordinator Group Interval (Steps)", 1, 10, 4, 1
            ), "microprovider_coordinator_group_interval": st.number_input(
                "Microprovider Coordinator Group Interval (Steps)", 1, 10, 4, 1
            ),
                "p_micro_support_attendance": st.number_input(
                    "Probability of Microprovider Attending Coordinator Support Group", 0.0, 1.0, 0.01, 0.01
            ), "p_resident_support_attendance": st.number_input(
                    "Probability of Resident Attending Coordinator Support Group", 0.0, 1.0, 0.01, 0.01
            ),  "p_coord_microprovider_outreach": st.number_input(
                    "Probability of Coordinator Microprovider Outreach", 0.0, 1.0, 0.1, 0.01
            ), "p_promote_microproviding": st.number_input(
                    "Probability of Promoting Microproviding", 0.0, 1.0, 0.01, 0.01
            )
        }
        params.update(coordinator_advanced_params)

    # Add a button to run the simulation with confirmation and a warning modal
    def execute_simulation(run_params, warning_slot=None):
        with st.spinner("Running the care model simulation..."):
            run_params = run_params.copy()
            # Run the simulation with a coordinator
            run_params["n_coordinators"] = 1
            results_with_coordinator = run_care_model(run_params)

            # Run the simulation without a coordinator
            run_params["n_coordinators"] = 0
            results_without_coordinator = run_care_model(run_params)

            # Store results and data in session state for use in other pages
            st.session_state["results_with_coordinator"] = results_with_coordinator
            st.session_state["results_without_coordinator"] = results_without_coordinator

            data_with_coordinator, data_without_coordinator = prepare_visualization_data(
                results_with_coordinator, results_without_coordinator
            )

            st.session_state["data_with_coordinator"] = data_with_coordinator
            st.session_state["data_without_coordinator"] = data_without_coordinator

        # remove the warning placeholder when finished
        if warning_slot is not None:
            try:
                warning_slot.empty()
            except Exception:
                pass

        st.success("Simulation complete!")

    if st.button("Run Simulation", key="run_sim"):
        # Immediately show a warning in a removable placeholder and start the simulation
        warning_slot = st.empty()
        warning_slot.warning("Please stay on this page whilst the simulation runs — navigating away may interrupt execution.")
        execute_simulation(params, warning_slot)

elif page =="Model Results":

    if "results_with_coordinator" in st.session_state and\
        "results_without_coordinator" in st.session_state:

        model_results_markdown = """
        Now the simulation has been run, we can explore the results. This page
        explores data collected during the simulation, and visualises it in a way that allows us to
        compare the two.

        Where the data we have used for this simulation is based on some real-world statistics,
        the necessary behaviours and interactions between agents in the simulation are based on assumptions,
        and should be interpreted with caution.

        It is intended that the results of the simulation are used to generate discussion
        into how communities, micro-providers and residents might work together!
        """

        st.markdown(model_results_markdown)

        st.write("### Percentage of Residents Receiving Care Over Time")
        percent_receiving_care_markdown = """
        This plot aims to compare the percentages of residents receiving help from
        microproviders over time between the two simulations.
        """
        st.markdown(percent_receiving_care_markdown)

        percent_receiving_care = plot_percent_receiving_care(
            st.session_state["data_with_coordinator"],
            st.session_state["data_without_coordinator"]
        )
        st.pyplot(percent_receiving_care)

        agg_data_with_coordinator = aggregate_step_approaches_by_year(
            st.session_state["data_with_coordinator"])
        agg_data_without_coordinator = aggregate_step_approaches_by_year(
            st.session_state["data_without_coordinator"])

        st.write("### Approaches to Finding Care Over Time")
        approaches_markdown = """
        This plot aims to compare the total number of approaches made by 
        residents to micro-providers over time between the two simulations.
        """
        st.markdown(approaches_markdown)
        
        approaches_plot = approaches_plot(
            st.session_state["data_with_coordinator"],
            st.session_state["data_without_coordinator"])
        st.pyplot(approaches_plot)

        quality_over_time_fig = plot_quality_over_time(
            st.session_state["data_with_coordinator"],
            st.session_state["data_without_coordinator"]
        )

        st.write("### Average Quality of Care Over Time")
        
        appraoches_markdown = """
        This plot aims to compare the average quality of care received from microproviders
        over time between the two simulations. A microprovider's quality score is
        randomly calculated when it enters the model and increases when they interact
        with other microproviders or coordinators.
        """

        st.markdown(appraoches_markdown)

        st.pyplot(quality_over_time_fig)

        quality_threshold_over_time_fig = plot_quality_threshold_over_time(
            st.session_state["data_with_coordinator"],
            st.session_state["data_without_coordinator"]
        )

        st.write("### Percentage of Micro-Providers Above Quality Threshold Over Time")
        quality_threshold_markdown = """
        This plot aims to compare the standard at which residents hold their micro-providers
        when asking them to help support them with there care needs. For residents in each
        simulation this is a value that's randomly generated between 0 and 1 at the start of the simulation.
        For the simulation with a coordinatr, if residents engage with the coordinator, they will adopt the
        coordinators quality threshold rating, which is set as an advanced parameter before running the simulation.
        This is intended to give a representation of how residents in a community might hold micro-providers to different standards,
        and how this might change if there was a coordinator in the community who
        promoted micro-providers and encouraged residents to use them appropriately.
        """
        st.markdown(quality_threshold_markdown)
        st.pyplot(quality_threshold_over_time_fig)

        plot_step_based_approaches_by_year_fig = plot_step_based_approaches_by_year(
           agg_data_with_coordinator,
            agg_data_without_coordinator
        )

        st.write("### Step-Based Approaches by Year")
        step_based_approaches_markdown = """
        This plot aims to compare the number of approaches made by residents to micro-providers by
        their constituent types (i.e., random approaches, coordinator approaches,
        and recommendations from other residents or microproviders based on their engagement with
        other agents.
        """
        st.markdown(step_based_approaches_markdown)
        st.pyplot(plot_step_based_approaches_by_year_fig)

    else:
        # Display a message if the simulation hasn't been run
        st.write("Please run the simulation first and then return to this page!")

                
elif page == "Network Analysis":

    # Check if the simulation has been run
    if "results_with_coordinator" in st.session_state and\
        "results_without_coordinator" in st.session_state:
        
        st.header("Network Analysis")

        network_analysis_markdown = """
        This page visualises provides an insight into the networks that form between
        residents, micro-providers and coordinators in the simulation. You can use the tab
        functionality to compare the networks that form in the simulation with a coordinator, and without a coordinator.
        """

        st.markdown(network_analysis_markdown)

        tab1, tab2 = st.tabs(["With Coordinator", "Without Coordinator"])

        with tab1:
            st.subheader("Network Analysis (With Coordinator)")
            results_with_coordinator = st.session_state["results_with_coordinator"]
            G, pos, coordinators, microproviders, residents = create_network_graph(
                results_with_coordinator["data_coord_registry"],
                results_with_coordinator["data_microprovider_registry"],
                results_with_coordinator["data_resident_registry"]
            )
            fig = visualize_network(G, pos, coordinators, microproviders, residents)
            st.pyplot(fig)

        with tab2:
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
        st.write("Please run the simulation first and then return to this page!")

elif page == "Do You Dig it?":
    # Initialize or load comments
    COMMENTS_FILE = "comments.csv"

    def load_comments():
        if not os.path.exists(COMMENTS_FILE):
            return []
        try:
            df = pd.read_csv(COMMENTS_FILE)
        except pd.errors.EmptyDataError:
            return []

        # If 'comment' column exists, return its values; otherwise try reading
        # the file as a single-column CSV (no header)
        if "comment" in df.columns:
            return df["comment"].astype(str).tolist()
        try:
            df2 = pd.read_csv(COMMENTS_FILE, header=None, names=["comment"]) 
            return df2["comment"].astype(str).tolist()
        except Exception:
            return []

    def save_comment(comment):
        comments = load_comments()
        comments.append(comment)
        pd.DataFrame({"comment": comments}).to_csv(COMMENTS_FILE, index=False)

    # Comments summary helper removed.

    # Streamlit App
    st.subheader("Do you Dig it? Share Your Thoughts!")

    COUNTS_FILE = "dig_counts.json"

    def load_counts():
        if not os.path.exists(COUNTS_FILE):
            return {"dig": 0, "dont_dig": 0}
        try:
            with open(COUNTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"dig": int(data.get("dig", 0)), "dont_dig": int(data.get("dont_dig", 0))}
        except Exception:
            return {"dig": 0, "dont_dig": 0}

    def save_counts(counts):
        try:
            with open(COUNTS_FILE, "w", encoding="utf-8") as f:
                json.dump({"dig": int(counts.get("dig", 0)), "dont_dig": int(counts.get("dont_dig", 0))}, f)
        except Exception:
            pass

    counts = load_counts()

    # Ensure session flags exist so a user can only vote once per session
    if "voted" not in st.session_state:
        st.session_state.voted = False
        st.session_state.voted_choice = None

    if not st.session_state.voted:
        col1, col2 = st.columns([0.1, 0.9])
        if col1.button("I dig it!"):
            counts["dig"] += 1
            save_counts(counts)
            st.session_state.voted = True
            st.session_state.voted_choice = "dig"
            st.success(f"Yay—we're out of shovels, take a spoon! :spoon: — Total who've selected dig it: {counts['dig']}")

        if (not st.session_state.voted) and col2.button("I don't dig it!"):
            counts["dont_dig"] += 1
            save_counts(counts)
            st.session_state.voted = True
            st.session_state.voted_choice = "dont_dig"
            st.success(f"That's fine!—Thanks for taking the time to look through the app! — Total who've selected don't dig it: {counts['dont_dig']}")
    else:
        # User already voted this session — show confirmation with totals
        if st.session_state.voted_choice == "dig":
            st.success(f"You already selected 'I dig it!' — Total who've selected dig it: {counts['dig']}")
        elif st.session_state.voted_choice == "dont_dig":
            st.success(f"You already selected 'I don't dig it!' — Total who've selected don't dig it: {counts['dont_dig']}")
    # Comment Input
    st.subheader("Upload some feedback to the web app (anonymous feedback welcome!)")
    st.info("If you'd like to get in touch via email instead, you can email\
            james.hough@somerset.gov.uk")
    comment = st.text_area("Your Comment", placeholder="Type your thoughts here...")
    if st.button("Submit Comment"):
        if comment.strip():
            save_comment(comment)
            st.success("Thank you for your comment!")
        else:
            st.error("Comment cannot be empty.")

    # Comments visualization and raw display
    comments = load_comments()
    st.write(pd.DataFrame({"comment": comments}))

elif page == "Thank Yous, Community Groups in Somerset, and Get in Touch!":
    st.header("Thank Yous, Community Groups in Somerset, and Get in Touch!")
    thank_you_markdown = """
    I'd like to extend a huge thank you to everyone who has taken to everyone
    involved in the creation and execution of the capacity building programmes
    (especially the wonderful PenARC!) which have provided the time, space, patience
    and expertise to learn about coding, agent based simulations and how to apply it
    to faucets of my own work with micro-providers!

    Also, a big thank you to Somerset Council and colleagues for advice and 
    continued support in allowing me to pursue this opportunity!

    The community groups we drew inspiration from for this simulation are thriving
    in Somerset, below are some examples which you should explore if you're interested
    in learning more about community-led support!
    - [Wincanton Cares](https://www.wincantoncares.org)
    - [Wivey Cares](https://www.wiveycares.net)
    - [Chard Community Hub](https://www.chardcommunityhub.com/)
    - [Blackdown Support Group](https://www.blackdownsupportgroup.org.uk/)
    - [Wells Community Network](https://www.wellscommunity.network/)
    - [Porlock Vale CIC](https://porlockvalecic.org/)
    - [Taunton Trusted Providers](https://www.tauntontrustedproviders.co.uk/)

    And finally, if you would like to contact me around this project my email is:
    james.hough@somerset.gov.uk
    """

    st.markdown(thank_you_markdown)

    