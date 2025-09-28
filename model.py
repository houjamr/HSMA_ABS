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

# ABM
class Resident_Agent(Agent):
    # constructor
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # residents unpaidcarers -- not used at the moment
        self.unpaidcarers = []
        # residents unpaidcare received -- not used at the moment
        self.unpaidcare_rec = []
        # residents microproviders
        self.microproviders = []
        # residents care received from microproviders
        self.packages_of_care_received = []
        # residents who have reviewed the care received by a micro and would 
        # promote them to other residents
        self.microproviders_to_recommend = []
        # residents who have attempted to contract a microprovider but have not 
        # met quality standard, therefore not contracted and not promoted
        self.blacklisted_microproviders = []
        # residents care needs met
        self.care_needs_met = False
        # initate residents care needs self.care_needs
        self.initiate_care_needs()
        # initiate micro quality threshold self.micro_quality_threshold
        self.initiate_micro_quality_threshold()

    # functions to initiate care needs
    def initiate_care_needs(self):
        self.generate_care_needs = round(np.random.normal(
            self.model.resident_care_need_max,
            self.model.resident_care_need_min
            ),
            1
        )
        self.care_needs = self.generate_care_needs
        
    # function to initiate micro quality threshold
    def initiate_micro_quality_threshold(self):
        self.generate_mic_qual_thrshld = round(random.uniform(0,1),3)
        self.micro_quality_threshold = self.generate_mic_qual_thrshld
    
    # function to decide if care needs are met
    def decide_needs_met(self):
        concat_list = self.packages_of_care_received + self.unpaidcare_rec
        if self.care_needs <= sum(concat_list):
            self.care_needs_met = True
        else:
            self.care_needs_met = False

    # randomly approaching a microprovider
    def contract_random_microprovider(self):
        """Attempt to contract with random provider."""
        if not self.model.microprovider_agent_registry:
            return False
            
        available_providers = [
            mp_id for mp_id in self.model.microprovider_agent_registry.keys()
            if mp_id not in self.blacklisted_microproviders]
        
        if not available_providers:
            return False
            
        random_microprovider_id = random.choice(available_providers)
        random_microprovider = self.model.microprovider_agent_registry[
            random_microprovider_id]['agent_object']
        
        if self.contract_microprovider(random_microprovider_id,
                                        random_microprovider):
            self.model.num_micros_approached_randomly += 1
            self.model.logger.info(
                f"Resident {self.unique_id} "
                f"contracted Microprovider {random_microprovider_id} "
                "after randomly approaching them")
            return True
        return False
   
    # unpaidcarers recommending a microprovider to their assigned resident
    # this might not do anything at the moment -- not used at the moment
    def check_unpaidcarers_microproviders(self):
        """Check for and attempt to contract a microprovider recommended by unpaid carers.
        
        Gets recommendations from resident's unpaid carers and attempts to contract
        one of their recommended providers.
        
        Returns:
            bool: True if successfully contracted a microprovider, False otherwise
        """
        # Get all recommendations from unpaid carers
        carer_micro_list = self._get_carer_recommendations()
        
        # Filter out already contracted providers
        possible_micros = [
            mp for mp in carer_micro_list 
            if mp not in self.microproviders
        ]
        
        if not possible_micros:
            return False

        # Select and attempt to contract random recommended provider
        recommended_provider_id = random.choice(possible_micros)
        recommended_provider = self.model.microprovider_agent_registry[
            recommended_provider_id]['agent_object']
        
        if self.contract_microprovider(recommended_provider_id, recommended_provider):
            self.model.num_micros_approached_carer_recommended += 1
            return True
            
        return False

    # get carer recommendations -- not used at the moment
    def _get_carer_recommendations(self):
        carer_micro_list = []
        for carer_id in self.unpaidcarers:
            carer = self.model.unpaidcarer_agent_registry[carer_id]
            recommended_micros = carer['microproviders_to_recommend']
            if recommended_micros:
                carer_micro_list.extend(recommended_micros)
                
        return carer_micro_list

    # check a residents list of recommended microproviders.
    def check_recommended_microproviders(self):
        # Filter for recommended microproviders that still exist and aren't contracted
        possible_micros = [
            mp for mp in self.microproviders_to_recommend 
            if (mp not in self.microproviders and 
                mp in self.model.microprovider_agent_registry)
        ]
        
        if not possible_micros:
            # Clean up recommendations list by removing non-existent providers
            self.microproviders_to_recommend = [
                mp for mp in self.microproviders_to_recommend
                if mp in self.model.microprovider_agent_registry
            ]
            return False

        # Select and attempt to contract random recommended provider
        recommended_provider_id = random.choice(possible_micros)
        recommended_provider = self.model.microprovider_agent_registry[
            recommended_provider_id]['agent_object']
        
        if self.contract_microprovider(recommended_provider_id,
                                        recommended_provider):
            self.model.num_micros_approached_recommended += 1
            self.model.logger.info(f"Resident {self.unique_id} contracted "
                                   f"Microprovider {recommended_provider_id} " 
                                   "based on recommendation from others")
            return True
            
        return False
    
    # checks if the model running with an active coordinator?
    def has_active_coordinator(self):
        if self.model.step_count < 0:  # Check step count
            return False
    
        return (self.model.num_coordinator_agents > 0 and 
                self.model.coordinator_agent_registry[0]\
                ['registered_microproviders'])

    # working with coordinator to find a microprovider
    def get_eligible_microproviders(self):
        eligible = []
        registered_providers = self.model.coordinator_agent_registry[0]\
        ['registered_microproviders']
        
        for micro_id in registered_providers:
            micro = self.model.microprovider_agent_registry[micro_id]\
            ['agent_object']
            if (micro.has_capacity and 
            micro.micro_quality >= self.micro_quality_threshold):
                eligible.append(micro_id)
                
        return eligible
    
    # resident finding a package of care through a coordinator
    def coord_care_brokerage(self):
        """Attempts to find and contract through coordinator."""
        if not self.has_active_coordinator():
            return False
            
        # Get only existing registered providers
        registered_providers = [
            mp for mp in self.model.coordinator_agent_registry[0]\
            ['registered_microproviders']\
            if mp in self.model.microprovider_agent_registry
        ]
        
        # Update coordinator registry to remove non-existent providers
        self.model.coordinator_agent_registry[0]\
        ['registered_microproviders'] = registered_providers
        
        eligible_microproviders = []
        for micro_id in registered_providers:
            micro = self.model.microprovider_agent_registry[micro_id]\
            ['agent_object']
            if (micro.has_capacity and 
            micro.micro_quality >= self.micro_quality_threshold):
                eligible_microproviders.append(micro_id)
                
        if not eligible_microproviders:
            return False
            
        if self.attempt_contract_with_eligible(eligible_microproviders):
            self.model.num_micros_approached_coordinator += 1
            self.model.logger.info(f"Resident {self.unique_id} contracted "
                                   f"Microprovider {micro_id} through a "
                                   "coordinator")
            return True
        return False
    
    # attempt to contract with a microprovider based on functions above if there
    # is more than one potential microprovider (i.e., not approaching one
    # randomly)
    def attempt_contract_with_eligible(self, eligible_microproviders):
        random.shuffle(eligible_microproviders)
        
        for micro_id in eligible_microproviders:
            micro = self.model.microprovider_agent_registry[micro_id]\
            ['agent_object']
            if self.contract_microprovider(micro_id, micro):
                return True
                
        return False

    # function for actually contracting the microprovider (used inside of 
    # attempt to contract with eligible micro_provider)
    def contract_microprovider(self, micro_id, micro_provider):
        # Early returns for invalid conditions
        if micro_id in self.blacklisted_microproviders:
            return False
            
        if micro_provider.micro_quality < self.micro_quality_threshold:
            return False
            
        if not micro_provider.has_capacity:
            return False
            
        if micro_id in self.microproviders:
            return False
            
        # Calculate care package size
        care_delivered = self._calculate_care_package(micro_id)
        
        self._update_resident_records(micro_provider, care_delivered)
        
        self._update_microprovider_records(micro_provider, care_delivered)
        
        return True

    # function used in contract_microprovider to calculate the care package
    def _calculate_care_package(self, micro_id):
        max_capacity = self.model.microprovider_agent_registry[micro_id]\
        ['agent_care_capacity']

        return round(random.uniform(1, max_capacity), 1)

    # function to update resident records after contracting a microprovider
    def _update_resident_records(self, micro_provider, care_delivered):
        self.microproviders.append(micro_provider.unique_id)
        self.packages_of_care_received.append(care_delivered)
        self.decide_needs_met()

    # function to update microprovider records after contracting a microprovider
    def _update_microprovider_records(self, micro_provider, care_delivered):
        micro_provider.residents.append(self.unique_id)
        micro_provider.packages_of_care.append(care_delivered)
        micro_provider.decide_capacity()
    
    # process for reviewing care received from microproviders -- how residents
    # come to influence the microprovider market
    def review_care_received(self):
        for index, micro in enumerate(self.microproviders):
            
            micro_object = self.model.microprovider_agent_registry[micro]\
                ['agent_object']
            
            if self._is_satisfied_with_care(micro_object):
                self._handle_satisfactory_care(micro)
            else:
                self._handle_unsatisfactory_care(index, micro, micro_object)

    def _is_satisfied_with_care(self, micro_object):
        return self.micro_quality_threshold <= micro_object.micro_quality

    def _handle_satisfactory_care(self, micro):
        if random.choice([True, False]) and\
        micro not in self.microproviders_to_recommend:
            self.microproviders_to_recommend.append(micro)
            self.model.logger.info(
                f"Resident {self.unique_id} is satisfied with care from "
                f"Microprovider {micro} and will recommend them to others"
            )

    def _handle_unsatisfactory_care(self, index, micro, micro_object):
        """Handles case where resident is unsatisfied with care.
        
        Removes care relationship and updates all relevant records.
        
        Args:
            index: Index of microprovider in resident's lists
            micro: ID of microprovider being reviewed
            micro_object: MicroProvider_Agent object
        """
        # Update resident records
        self.blacklisted_microproviders.append(micro)
        pckg_t_rmv = self.packages_of_care_received[index]
        self.packages_of_care_received.remove(pckg_t_rmv)
        self.microproviders.remove(micro)
        
        if micro in self.microproviders_to_recommend:
            self.microproviders_to_recommend.remove(micro)

        # Update microprovider records
        resi_index = micro_object.residents.index(self.unique_id)
        resi_package = micro_object.packages_of_care[resi_index]
        micro_object.residents.remove(self.unique_id)
        micro_object.packages_of_care.remove(resi_package)
        self.model.logger.info(
            f"Resident {self.unique_id} is unsatisfied with care from "
            f"Microprovider {micro} and has ended the care relationship"
        )

    def promote_microprovider(self):
        """
        Promote recommended microproviders to nearby residents.
        
        Finds nearby residents in the same cell and shares microprovider 
        recommendations with them if they haven't blacklisted those providers.
        """
        if not self.microproviders_to_recommend:
            return
        
        # Get nearby residents
        possible_residents = [
        rid for rid in self.model.resident_agent_registry.keys() 
        if rid != self.unique_id]
    
        if not possible_residents:
            return
        
        # Choose random resident to chat with
        resident_id = random.choice(possible_residents)
        resident_to_chat = self.model.resident_agent_registry[resident_id]\
        ['agent_object']

        num_to_share = min(
            random.randint(1, 2), len(self.microproviders_to_recommend))
        selected_micros = random.sample(
            self.microproviders_to_recommend, num_to_share)
        
        # Share recommendations
        for micro in selected_micros:
            if (micro not in resident_to_chat.microproviders_to_recommend and 
                micro not in resident_to_chat.blacklisted_microproviders):
                resident_to_chat.microproviders_to_recommend.append(micro)
                self.model.logger.info(
                f"Resident {self.unique_id} promoted Microprovider {micro} "
                f"to Resident {resident_to_chat.unique_id}")
        
    def _try_find_care(self):
            """Attempt to find care through various channels in priority order.
            
            Returns:
                bool: True if care was found, False otherwise
            """
            # Try recommended providers first
            if self.microproviders_to_recommend:
                if self.check_recommended_microproviders():
                    return True
                
            # Try unpaid carer recommendations next
            if self.check_unpaidcarers_microproviders():
                return True
                
            # Small chance to use coordinator
            if random.random() < self.model.p_use_coordinator:
                if self.coord_care_brokerage():
                    return True
                
            # Otherwise try random provider
            elif random.random() < self.model.p_approach_random_micro:
                if self.contract_random_microprovider():
                    return True
            return False

    def _periodic_care_review(self):
        """Periodically review and promote care arrangements."""
        # 5% chance to review current care
        if random.random() < self.model.p_review_care:
            self.review_care_received()
        
        # 5% chance to promote providers
        if random.random() < self.model.p_promote_micro:
            self.promote_microprovider()
 
        self.decide_needs_met()
        
    def step(self):
        self.decide_needs_met()
        
        if not self.care_needs_met:
            if self._try_find_care():
                return
                
        self._periodic_care_review()

 #MicroProvider Agent   

class MicroProvider_Agent(Agent):
    # constructor
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.residents = []
        self.packages_of_care = []
        self.has_capacity = True
        self.initiate_care_capacity()
        self.initiate_micro_quality_score()

    def initiate_care_capacity(self):
        self.generate_care_capacity = round(random.uniform(10,50),1)
        self.care_capacity = self.generate_care_capacity

    def initiate_micro_quality_score(self):
        self.generate_mic_qual = round(random.uniform(0,1),3)
        self.micro_quality = self.generate_mic_qual

    def decide_capacity(self):
        if sum(self.packages_of_care) > self.care_capacity:
            self.has_capacity = False
        else:
            self.has_capacity = True

    def register_with_coordinator(self):
        """Attempt to register microprovider with coordinator if eligible.
        Registers with coordinator if:
        1. Coordinator exists
        2. Not already registered
        3. Has at least one resident
        4. Meets quality threshold
        """
        if not self.model.num_coordinator_agents:
            return
            
        coordinator_registry = self.model.coordinator_agent_registry[0]
        
        if (self.unique_id not in coordinator_registry\
        ['registered_microproviders'] and len(self.residents) >= 1 and
        self.micro_quality >= coordinator_registry['micro_quality_threshold']):
            
            coordinator_registry['registered_microproviders'].\
                append(self.unique_id)
            self.model.logger.info(f"Microprovider {self.unique_id} registered "
                                   "with the Coordinator")

    def step(self):
        self.decide_capacity()

        if self.model.num_coordinator_agents > 0 and\
            self.unique_id not in self.model.coordinator_agent_registry[0]\
                ['registered_microproviders']:
            if random.random() < self.model.micro_join_coord:
                self.register_with_coordinator()

'''
Unpaid Care Agent not used
'''
class UnpaidCare_Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.residents = []
        self.unpaidcare_delivered = []
        self.microproviders_to_recommend = []
        self.blacklisted_microproviders = []
        self.has_capacity = True
        self.initiate_care_capacity()

    def initiate_care_capacity(self):
        self.generate_care_capacity = round(random.uniform(5,10),1)
        self.care_capacity = self.generate_care_capacity

    def decide_capacity(self):
        if sum(self.unpaidcare_delivered) > self.care_capacity:
            self.has_capacity = False
        else:
            self.has_capacity = True

    def locate_resident(self):
        """Attempt to find and provide care to a random resident.
        
        Selects a random resident and if the carer has capacity and isn't already 
        caring for them, establishes a care relationship.
        """
        # Select random resident
        allocated_resident_id = random.choice(
            list(self.model.resident_agent_registry.keys())
        )
        
        allocated_resident = self.model.resident_agent_registry[
            allocated_resident_id]['agent_object']

        # Check if can provide care
        if not (self.has_capacity and 
                self.unique_id not in allocated_resident.unpaidcarers):
            return
            
        # Calculate care amount
        care_delivered = round(
            random.uniform(1, self.care_capacity),
            1
        )
        
        # Update carer records
        self._update_carer_records(allocated_resident.unique_id, care_delivered)
        
        # Update resident records
        self._update_resident_records(allocated_resident, care_delivered)

    def _update_carer_records(self, resident_id, care_delivered):
        """Update carer's records with new care arrangement.
        
        Args:
            resident_id: ID of resident receiving care
            care_delivered: Amount of care to be delivered
        """
        self.residents.append(resident_id)
        self.unpaidcare_delivered.append(care_delivered)
        self.decide_capacity()

    def _update_resident_records(self, resident, care_delivered):
        """Update resident's records with new care arrangement.
        
        Args:
            resident: Resident_Agent object receiving care
            care_delivered: Amount of care to be delivered
        """
        resident.unpaidcarers.append(self.unique_id)
        resident.unpaidcare_rec.append(care_delivered)
        resident.decide_needs_met()

    def fetch_microproviders(self):
        if len(self.residents) > 0:
            for resident_id in self.residents:
                resident_object = self.model.resident_agent_registry\
                    [resident_id]['agent_object']
                
                # Get the microproviders this resident recommends
                resident_recommended_microproviders = resident_object\
                    .microproviders_to_recommend
                
                # Add recommended microproviders to unpaid carer's list
                # if not already there
                for micro_id in resident_recommended_microproviders:
                    if micro_id not in self.microproviders_to_recommend:
                        self.microproviders_to_recommend.append(micro_id)

    def promote_microprovider(self):
        """Promote recommended microproviders to nearby agents.
        
        Finds nearby residents and unpaid carers in the same cell and shares 
        microprovider recommendations with them if they haven't blacklisted 
        those providers.
        """
        # Get nearby agents that can receive recommendations
        nearby_agents = self._get_nearby_recommendation_targets()
        
        if not nearby_agents or not self.microproviders_to_recommend:
            return
            
        # Choose a random agent to share recommendations with
        target_agent = random.choice(nearby_agents)
        
        # Share recommendations
        self._share_recommendations(target_agent)

    def _get_nearby_recommendation_targets(self):
        """Get list of nearby agents that can receive recommendations.
        
        Returns:
            list: Agents in same cell that can receive recommendations
        """
        nearby_agents = []
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        
        for inhabitant in cellmates:
            if isinstance(inhabitant, (Resident_Agent, UnpaidCare_Agent)):
                nearby_agents.append(inhabitant)
                
        return nearby_agents

    def _share_recommendations(self, target_agent):
        """Share microprovider recommendations with target agent.
        
        Args:
            target_agent: Agent to receive recommendations
        """
        for micro in self.microproviders_to_recommend:
            if self._can_recommend_to_agent(micro, target_agent):
                target_agent.microproviders_to_recommend.append(micro)

    def _can_recommend_to_agent(self, micro, agent):
        """Check if microprovider can be recommended to agent.
        
        Args:
            micro: ID of microprovider to recommend
            agent: Agent to receive recommendation
            
        Returns:
            bool: True if provider can be recommended, False otherwise
        """
        return (micro not in agent.microproviders_to_recommend and
                micro not in agent.blacklisted_microproviders)
    
    def step(self):
        if self.has_capacity:
            self.locate_resident()
        self.fetch_microproviders()

        if random.choice([True, False]):
            self.promote_microprovider()

class Coordinator_Agent(Agent):
    # constructor
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.registered_microproviders = []
        self.micro_quality_threshold = self.model.micro_quality_threshold

    def step(self):
        pass

class Care_Model(Model):
    def __init__(self,
                 N_RESIDENT_AGENTS,
                 N_MICROPROVIDER_AGENTS,
                 N_UNPAIDCARE_AGENTS,
                 N_COORDINATOR_AGENTS,
                 width,
                 height,
                 resident_care_need_min,
                 resident_care_need_max,
                 microprovider_care_cap_min,
                 microprovider_care_cap_max,
                 p_resident_join,
                 p_microprovider_join,
                 p_resident_leave,
                 p_microprovider_leave,
                 p_use_coordinator,
                 p_approach_random_micro,
                 p_review_care,
                 p_promote_micro,
                 micro_quality_threshold,
                 micro_join_coord,
                 random_seed=None):  
        super().__init__()

        self.random_seed = random_seed
        # Set the random seed for reproducibility
        if random_seed is not None:
            self.random = random.Random(random_seed)
        else:
            self.random = random.Random()

        # Initialize other model attributes here

        # Create a buffer to hold log messages
        self.log_buffer = StringIO()

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a handler that writes to the buffer
        buffer_handler = logging.StreamHandler(self.log_buffer)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        buffer_handler.setFormatter(formatter)

        # Remove any existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create a handler that writes to the buffer
        buffer_handler = logging.StreamHandler(self.log_buffer)
        formatter = logging.Formatter('%(message)s')
        buffer_handler.setFormatter(formatter)
        self.logger.addHandler(buffer_handler)

        self.logger.propagate = False

        # model variables
        self.num_resident_agents = N_RESIDENT_AGENTS
        self.num_microprovider_agents = N_MICROPROVIDER_AGENTS
        self.num_unpaidcare_agents = N_UNPAIDCARE_AGENTS
        self.num_coordinator_agents = N_COORDINATOR_AGENTS
        self.running = True
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.step_count = 0

        # variables for probabilities and thresholds
        self.resident_care_need_min = resident_care_need_min
        self.resident_care_need_max = resident_care_need_max
        self.microprovider_care_cap_min = microprovider_care_cap_min
        self.microprovider_care_cap_max = microprovider_care_cap_max
        self.p_resident_join = p_resident_join
        self.p_microprovider_join = p_microprovider_join
        self.p_resident_leave = p_resident_leave
        self.p_microprovider_leave = p_microprovider_leave
        self.p_use_coordinator = p_use_coordinator
        self.p_approach_random_micro = p_approach_random_micro
        self.p_review_care = p_review_care
        self.p_promote_micro = p_promote_micro
        self.micro_quality_threshold = micro_quality_threshold
        self.micro_join_coord = micro_join_coord

        # counter of number of micros approached
        self.num_micros_approached_randomly = 0
        self.num_micros_approached_recommended = 0
        self.num_micros_approached_carer_recommended = 0
        self.num_micros_approached_coordinator = 0
        #initialise agent registries
        self.resident_agent_registry = {}
        self.microprovider_agent_registry = {}
        self.unpaidcarer_agent_registry = {}
        self.coordinator_agent_registry = {}

        # adding resident agents
        for i in range(self.num_resident_agents):
            a = Resident_Agent(i, self)
            self.schedule.add(a)
            self.resident_agent_registry[i] = {
                'agent_object' : a,
                'agent_id': i,
                'care_needs_met': False,
                'agent_care_needs': a.care_needs,
                'micro_quality_threshold': a.micro_quality_threshold,
                'allocated_microproviders': [],
                'packages_of_care_received': [],
                'microproviders_to_recommend': [],
                'blacklisted_microproviders': [],
                'unpaidcarers': [],
                'unpaidcare_rec': []
            }
           
            try:
                start_cell = self.grid.find_empty()
                self.grid.place_agent(a, start_cell)
            except:
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                self.grid.place_agent(a, (x,y))
            # print(f'placed resident agent {a.pos}')

        # adding micro-provider agents
        for i in range(self.num_microprovider_agents):
            a = MicroProvider_Agent(i, self)
            self.schedule.add(a)
            self.microprovider_agent_registry[i] = {
                'agent_object': a,
                'agent_id': i,
                'agent_care_capacity' : a.care_capacity,
                'micro_quality': a.micro_quality,
                'has_capacity': True,
                'allocated_residents': [],
                'packages_of_care_delivered': []
            }

            try:
                start_cell = self.grid.find_empty()
                self.grid.place_agent(a, start_cell)
            except:
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                self.grid.place_agent(a, (x,y))
         
        # adding unpaid carer agents
        for i in range(self.num_unpaidcare_agents):
            a = UnpaidCare_Agent(i, self)
            self.schedule.add(a)
            self.unpaidcarer_agent_registry[i] = {
                'agent_object': a,
                'agent_id': i,
                'agent_care_capacity': a.generate_care_capacity,
                'has_capacity': a.has_capacity,
                'residents': [],
                'unpaidcare_delivered': [],
                'microproviders_to_recommend': []
            }

            try:
                start_cell = self.grid.find_empty()
                self.grid.place_agent(a, start_cell)
            except:
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                self.grid.place_agent(a, (x,y))

        # adding coordinator agents
        for i in range(self.num_coordinator_agents):
            a = Coordinator_Agent(i, self)
            self.schedule.add(a)
            self.coordinator_agent_registry[i] = {
                'agent_object': a,
                'agent_id': i,
                'registered_microproviders': [],
                'micro_quality_threshold': a.micro_quality_threshold
            }

        # adding datacollector to model
        self.datacollector = DataCollector(
            model_reporters={
                    "calc_allocated_residents":
                    self.calc_allocated_residents,
                    "calc_resident_packages":
                    self.calc_packages_of_care_delivered,
                    "calc_capacity":
                    self.calc_micro_capacity,
                    "has_capacity":
                    self.micro_has_capacity,
                    "number_of_micros_randomly_approached":
                    self.calc_num_micros_approached_randomly,
                    "number_of_micros_recommended":
                    self.calc_num_micros_approached_recommended,
                    "number_of_micros_carer_recommended":
                    self.calc_num_micros_approached_carer_recommended,
                    "number_of_micros_approached_coordinator":
                    self.calc_num_micros_approached_coordinator,
                    "calc_is_receiving_care":
                    self.calc_receiving_care,}
        )

    # Add these new methods to access the logs
    def print_logs(self):
        """Print all accumulated log messages."""
        self.log_buffer.seek(0)
        print(self.log_buffer.read())
        
    def clear_logs(self):
        """Clear all accumulated log messages."""
        self.log_buffer.truncate(0)
        self.log_buffer.seek(0)

    # adding agents as model runs
    def add_new_agents(self):
        """
        Add new agents to the model with small random chance.
        """
        # Chance for new resident to join
        if random.random() < 0.01:
            new_id = max(self.resident_agent_registry.keys()) + 1
            self._add_new_resident(new_id)
            
        # Chance for new microprovider to join    
        if random.random() < 0.002:
            new_id = max(self.microprovider_agent_registry.keys()) + 1
            self._add_new_microprovider(new_id)
        
    def _add_new_resident(self, new_id):
        a = Resident_Agent(new_id, self)
        self.schedule.add(a)
        self.resident_agent_registry[new_id] = {
            'agent_object': a,
            'agent_id': new_id,
            'care_needs_met': False,
            'agent_care_needs': a.care_needs,
            'micro_quality_threshold': a.micro_quality_threshold,
            'allocated_microproviders': [],
            'packages_of_care_received': [],
            'microproviders_to_recommend': [],
            'blacklisted_microproviders': [],
            'unpaidcarers': [],
            'unpaidcare_rec': []
        }
        
        try:
            start_cell = self.grid.find_empty()
            self.grid.place_agent(a, start_cell)
        except:
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
        
        self.num_resident_agents += 1
        self.logger.info(
            f"New Resident {new_id} joined with {a.care_needs} care needs")
        
    def _add_new_microprovider(self, new_id):
        a = MicroProvider_Agent(new_id, self)
        self.schedule.add(a)
        self.microprovider_agent_registry[new_id] = {
            'agent_object': a,
            'agent_id': new_id,
            'agent_care_capacity': a.care_capacity,
            'micro_quality': a.micro_quality,
            'has_capacity': True,
            'allocated_residents': [],
            'packages_of_care_delivered': []
        }
        
        try:
            start_cell = self.grid.find_empty()
            self.grid.place_agent(a, start_cell)
        except:
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
            
        self.num_microprovider_agents += 1
        self.logger.info(
            f"New Microprovider {new_id} joined with {a.care_capacity}")
     
    # removing agents as model runs
    def remove_agent(self, agent): 
        self.schedule.remove(agent)
        self.grid.remove_agent(agent)
        
        # Update registries based on agent type
        if isinstance(agent, Resident_Agent):
            # Remove resident from all microproviders
            for micro_id in agent.microproviders:
                micro = self.microprovider_agent_registry[micro_id]['agent_object']
                if agent.unique_id in micro.residents:
                    idx = micro.residents.index(agent.unique_id)
                    micro.residents.remove(agent.unique_id)
                    micro.packages_of_care.pop(idx)
                    micro.decide_capacity()
                    
            # Remove from registry
            del self.resident_agent_registry[agent.unique_id]
            self.num_resident_agents -= 1
            
        elif isinstance(agent, MicroProvider_Agent):
            # Remove microprovider from all residents
            for resident_id in agent.residents:
                resident = self.resident_agent_registry[resident_id]
                ['agent_object']
                if agent.unique_id in resident.microproviders:
                    idx = resident.microproviders.index(agent.unique_id)
                    resident.microproviders.remove(agent.unique_id)
                    resident.packages_of_care_received.pop(idx)
                    resident.decide_needs_met()
                    
            # Remove from coordinator if registered
            if self.num_coordinator_agents > 0:
                if agent.unique_id in self.coordinator_agent_registry[0]\
                    ['registered_microproviders']:
                    self.coordinator_agent_registry[0]\
                    ['registered_microproviders'].remove(agent.unique_id)
                    
            # Remove from registry
            del self.microprovider_agent_registry[agent.unique_id]
            self.num_microprovider_agents -= 1

    def check_agent_removals(self):
        """Check and remove agents based on model conditions."""
        # Check residents
        for resident_id in list(self.resident_agent_registry.keys()):
            resident = self.resident_agent_registry[resident_id]['agent_object']
            if resident.care_needs_met and random.random() < self.p_resident_leave:
                self.remove_agent(resident)
                self.logger.info(
                    f"Resident {resident_id} has left the model")
                
        # Check microproviders
        for micro_id in list(self.microprovider_agent_registry.keys()):
            micro = self.microprovider_agent_registry[micro_id]['agent_object']
            if not micro.residents and random.random() < self.p_microprovider_leave:
                self.remove_agent(micro)
                self.logger.info(
                    f"Microprovider {micro_id} has left the model")

    def _update_agent_registry(self, agent):
        """Update the appropriate registry based on agent type."""
        if isinstance(agent, MicroProvider_Agent):
            self._update_microprovider_registry(agent)
        elif isinstance(agent, Resident_Agent):
            self._update_resident_registry(agent)
        elif isinstance(agent, UnpaidCare_Agent):
            self._update_unpaidcarer_registry(agent)
        elif isinstance(agent, Coordinator_Agent):
            self._update_coordinator_registry(agent)

    def _update_microprovider_registry(self, agent):
        """Update microprovider registry with current agent state."""
        registry = self.microprovider_agent_registry[agent.unique_id]
        registry.update({
            'allocated_residents': agent.residents,
            'packages_of_care_delivered': agent.packages_of_care,
            'has_capacity': agent.has_capacity
        })

    def _update_resident_registry(self, agent):
        """Update resident registry with current agent state."""
        registry = self.resident_agent_registry[agent.unique_id]
        registry.update({
            'allocated_microproviders': agent.microproviders,
            'packages_of_care_received': agent.packages_of_care_received,
            'care_needs_met': agent.care_needs_met,
            'blacklisted_microproviders': agent.blacklisted_microproviders,
            'microproviders_to_recommend': agent.microproviders_to_recommend,
            'unpaidcarers': agent.unpaidcarers,
            'unpaidcare_rec': agent.unpaidcare_rec
        })

    def _update_unpaidcarer_registry(self, agent):
        """Update unpaid carer registry with current agent state."""
        registry = self.unpaidcarer_agent_registry[agent.unique_id]
        registry.update({
            'residents': agent.residents,
            'unpaidcare_delivered': agent.unpaidcare_delivered,
            'has_capacity': agent.has_capacity,
            'microproviders_to_recommend': agent.microproviders_to_recommend
        })

    def _update_coordinator_registry(self, agent):
        """Update coordinator registry with current agent state."""
        registry = self.coordinator_agent_registry[agent.unique_id]
        registry.update({
            'registered_microproviders': agent.registered_microproviders
        })    
 
    # model functions for data collectors
    def calc_num_micros_approached_randomly(self):
        num = self.num_micros_approached_randomly
        return num
    
    def calc_num_micros_approached_recommended(self):
        num = self.num_micros_approached_recommended
        return num
    
    def calc_num_micros_approached_carer_recommended(self):
        num = self.num_micros_approached_carer_recommended
        return num
    
    def calc_num_micros_approached_coordinator(self):
        num = self.num_micros_approached_coordinator
        return num
    
    def calc_num_microproviders(self):
        """Calculate the total number of micro-providers in the model."""
        return len(self.microprovider_agent_registry)
    
    # microprovider functions for data collector
    def calc_allocated_residents(self):
        allocated_residents_total = []
        # Iterate through existing agents only
        for agent_id in self.microprovider_agent_registry:
            allocated_residents = self.microprovider_agent_registry[agent_id]\
                ['allocated_residents']
            if allocated_residents:
                allocated_residents_total.extend(allocated_residents)
        return len(allocated_residents_total)
    
    def calc_packages_of_care_delivered(self):
        packages_of_care_total = []
        # Iterate through existing agents only
        for agent_id in self.microprovider_agent_registry:
            packages_of_care = self.microprovider_agent_registry[agent_id\
                ]['packages_of_care_delivered']
            if packages_of_care:
                packages_of_care_total.extend(packages_of_care)
        return sum(packages_of_care_total)

    def calc_micro_capacity(self):
        total_care_capacity = []
        # Iterate through existing agents only
        for agent_id in self.microprovider_agent_registry:
            care_capacity = self.microprovider_agent_registry[agent_id]\
                ['agent_care_capacity']
            total_care_capacity.append(care_capacity)
        return sum(total_care_capacity)

    def micro_has_capacity(self):
        has_capacity = []
    # Iterate through existing agents only
        for agent_id in self.microprovider_agent_registry:
            if self.microprovider_agent_registry[agent_id]['has_capacity']:
                has_capacity.append(agent_id)
        return len(has_capacity)

    def calc_receiving_care(self):
        count_receiving_care = 0
        for agent_id in self.resident_agent_registry:
            if self.resident_agent_registry[agent_id]\
                ['packages_of_care_received'] != []:
                count_receiving_care += 1
        return count_receiving_care

    # model step
    def step(self):
        self.logger.info(f"Model step {self.step_count}")
        self.datacollector.collect(self)
        self.step_count += 1
        self.add_new_agents()
        self.check_agent_removals()
        
        # Update registries and step each agent
        for agent in self.schedule.agents:
            self._update_agent_registry(agent)
            agent.step()
            
        self.schedule.step() 

'''function to run the model
'''
def run_care_model(params=None):
    """
    Run the Care_Model simulation using parameters from a dictionary.

    Dynamically add micro-providers until the percentage of residents receiving
    help matches the HSE statistics.

    Parameters:
    -----------
    params : dict, optional
        A dictionary containing all the parameters required to initialize and run the model.
        If None, default values will be used.

    Returns:
    --------
    results : dict
        A dictionary containing the model, data, and agent registries.
    """
    # Use default values if params is None
    if params is None:
        params = {}

    # Initialize the model with a small number of micro-providers
    model = Care_Model(
        N_RESIDENT_AGENTS=params.get("n_residents", 833),
        N_MICROPROVIDER_AGENTS=params.get("n_microproviders", 1),  # Start with 1 micro-provider
        N_UNPAIDCARE_AGENTS=params.get("n_unpaidcarers", 0),
        N_COORDINATOR_AGENTS=params.get("n_coordinators", 0),
        width=params.get("width", 12),
        height=params.get("height", 12),
        resident_care_need_min=params.get("resident_care_need_min", 1),
        resident_care_need_max=params.get("resident_care_need_max", 20),
        microprovider_care_cap_min=params.get("microprovider_care_cap_min", 5),
        microprovider_care_cap_max=params.get("microprovider_care_cap_max", 40),
        # to be standardised
        p_resident_join=params.get("p_resident_join", 0.01),
        p_microprovider_join=params.get("p_microprovider_join", 0.001),
        p_resident_leave=params.get("p_resident_leave", 0.001),
        p_microprovider_leave=params.get("p_microprovider_leave", 0.001),
        # stay the same
        p_use_coordinator=params.get("p_use_coordinator", 0.02),
        p_approach_random_micro=params.get("p_approach_random_micro", 0.001),
        p_review_care=params.get("p_review_care", 0.2),
        p_promote_micro=params.get("p_promote_micro", 0.2),
        micro_quality_threshold=params.get("micro_quality_threshold", 0.5),
        micro_join_coord=params.get("micro_join_coord", 0.5),
        random_seed=params.get("random_seed", 42)
    )

    # Run the model for the specified number of steps
    n_steps = params.get("n_steps", 100)
    for i in range(n_steps):
        model.step()

        # Calculate the percentage of residents receiving help
        total_residents = len(model.resident_agent_registry)
        receiving_help = model.calc_receiving_care()
        receiving_help_percentage = receiving_help / total_residents * 100

        # Add micro-providers if the percentage is below the HSE statistic (18%)
        if receiving_help_percentage < 18:
            new_micro_id = max(model.microprovider_agent_registry.keys()) + 1
            model._add_new_microprovider(new_micro_id)

    # Collect data from the model
    data = model.datacollector.get_model_vars_dataframe()

    # Create resident registry DataFrame
    data_resident_registry = pd.DataFrame.from_dict(
        model.resident_agent_registry, orient="index"
    )
    data_resident_registry.drop(columns=["agent_object"], inplace=True)

    # Create microprovider registry DataFrame
    data_microprovider_registry = pd.DataFrame.from_dict(
        model.microprovider_agent_registry, orient="index"
    )
    data_microprovider_registry.drop(columns=["agent_object"], inplace=True)

    # Create coordinator registry DataFrame (if coordinators exist)
    if model.num_coordinator_agents > 0:
        data_coord_registry = pd.DataFrame.from_dict(
            model.coordinator_agent_registry, orient="index"
        )
        data_coord_registry.drop(columns=["agent_object"], inplace=True)
    else:
        data_coord_registry = pd.DataFrame()

    # Return all results as a dictionary
    return {
        "model": model,
        "data": data,
        "data_resident_registry": data_resident_registry,
        "data_microprovider_registry": data_microprovider_registry,
        "data_coord_registry": data_coord_registry
    }

params = {"n_steps": 1000}

results = run_care_model(params)

def plot(data):
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    data.plot(kind='line', y='number_of_micros_randomly_approached',ax=ax[0])
    data.plot(kind='line', y='number_of_micros_recommended',ax=ax[0])
    data.plot(kind='line', y='number_of_micros_approached_coordinator',ax=ax[0])
    data.plot(kind='line', y='has_capacity', ax=ax[1])
    data.plot(kind='line', y='calc_is_receiving_care', ax=ax[1])

    ax[0].legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
    ax[1].legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2)

    ax[0].set_title('Provider Approaches by Type')
    ax[1].set_title('System Capacity and Care Delivery')

    plt.tight_layout()
    return fig, ax

print(results['data'])
plot(results['data'])
plt.show()