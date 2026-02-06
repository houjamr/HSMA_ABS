from pyexpat import model
import mesa
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
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
        # initiate micro quality threshold
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
        random_microprovider = self.model.microprovider_agent_registry\
            [random_microprovider_id]['agent_object']
        
        if self.contract_microprovider(random_microprovider_id, random_microprovider):
            self.model.num_micros_approached_randomly += 1
            self.model.step_micros_approached_randomly += 1  # Increment step counter
            self.model.logger.info(
                f"Resident {self.unique_id} contracted Microprovider {random_microprovider_id} after randomly approaching them"
            )
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
            self.model.step_micros_approached_recommended += 1
            self.model.logger.info(f"Resident {self.unique_id} contracted "
                                   f"Microprovider {recommended_provider_id} " 
                                   "based on recommendation from others")
            return True
            
        return False
    
    # checks if the model running with an active coordinator?
    '''look at this again'''

    def has_active_coordinator(self):
        if self.model.step_count < 0:
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
            self.model.step_micros_approached_coordinator += 1  # Increment step counter
            self.model.logger.info(f"Resident {self.unique_id} contracted Microprovider {micro_id} through a coordinator")
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
        """
        Attempt to contract a microprovider. If the selected microprovider
        doesn't have capacity, try one of their peers.
        """
        # Early returns for invalid conditions
        if micro_id in self.blacklisted_microproviders:
            return False

        if micro_provider.micro_quality < self.micro_quality_threshold:
            return False

        if not micro_provider.has_capacity:
            self.model.logger.info(
                f"Microprovider {micro_id} has no capacity. Trying peers."
            )
            # Try peers of the microprovider
            random.shuffle(micro_provider.microprovider_peers)
            for peer_id in micro_provider.microprovider_peers:
                if peer_id not in self.blacklisted_microproviders:
                    peer = self.model.microprovider_agent_registry[peer_id]['agent_object']
                    if peer.has_capacity and peer.micro_quality >= self.micro_quality_threshold:
                        return self.contract_microprovider(peer_id, peer)
            return False

        if micro_id in self.microproviders:
            return False

        # Calculate care package size
        care_delivered = self._calculate_care_package(micro_id)

        self._update_resident_records(micro_provider, care_delivered)

        self._update_microprovider_records(micro_provider, care_delivered)

        return True
    
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
        Promote recommended microproviders o nearby residents.
        
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

        # Select a single random microprovider to recommend
        micro = random.choice(self.microproviders_to_recommend)
        micro_object = self.model.microprovider_agent_registry[micro]['agent_object']

        # Share the recommendation
        if (micro not in resident_to_chat.blacklisted_microproviders and
            micro not in resident_to_chat.microproviders_to_recommend and
            micro_object.micro_quality >= resident_to_chat.micro_quality_threshold):  # Check quality threshold
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

        if len(self.microproviders_to_recommend) > 0 and self.model.step_count % 20:
            self.microproviders_to_recommend.pop(0)

 #MicroProvider Agent   
class MicroProvider_Agent(Agent):
    # constructor
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.residents = []
        self.microprovider_peers = []
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
        if not self.model.num_coordinator_agents:
            return
            
        coordinator_registry = self.model.coordinator_agent_registry[0]
        
        if self.unique_id not in coordinator_registry['registered_microproviders']:
            if self.model.coordinator_has_threshold and\
            self.micro_quality >= coordinator_registry['coord_micro_quality_threshold']:
                coordinator_registry['registered_microproviders'].append(self.unique_id)
            if not self.model.coordinator_has_threshold:
                coordinator_registry['registered_microproviders'].append(self.unique_id)

    def step(self):
        self.decide_capacity()

        if random.random() < self.model.micro_join_coord:
                self.register_with_coordinator()

        if len(self.microprovider_peers) > 0 and self.model.step_count % 4:
            self.microprovider_peers.pop(0)
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
        self.coord_micro_quality_threshold = 0.5
        self.coordinator_has_threshold = self.model.coordinator_has_threshold

    # function that toggles if a coordinator uses the target quality threshold to
    #  decide whether to register a microprovider or not
        # microprovider function checks whether coordinator has quality threshold value enabled

    def microprovider_outreach(self):
        unregistered_microproviders = [mp for mp in self.model.microprovider_agent_registry.keys()
                                       if mp not in self.registered_microproviders]
        if unregistered_microproviders:
            target_microprovider_id = random.choice(unregistered_microproviders)
            target_microprovider = self.model.microprovider_agent_registry[target_microprovider_id]['agent_object']

            if self.coordinator_has_threshold:
                if target_microprovider.micro_quality >= self.coord_micro_quality_threshold:
                    self.registered_microproviders.append(target_microprovider.unique_id)
                else:
                    target_microprovider.micro_quality = min(target_microprovider.micro_quality + 0.05, 1.0)
            if not self.coordinator_has_threshold:
                self.registered_microproviders.append(target_microprovider.unique_id)
        else:
            self.model.logger.info("No unregistered microproviders available for outreach.")

    # function where a coordinator does outreach work with microproviders in the model
    # to encourage them to register/improve there quality rating
        # p chance coordintator does outreach work each step, if microprovider
        # passess threshold, recruit, otherwise improve rating by 0.05
        # (rating of a mp cannot exceed 1)
        
    # function where coordinator runs a microprovider peer support group
        # function where coordinator collects mps

    # function where coordinator runs a resident peer spport group to encourage
    # sharing of microprovider recommendations
        # collect n residents, residents share recommendations
        # if residents microprovider threshold is less than coorindator, add to
        # match but not exceed the threshold of the coordinator group.

    def run_microprovider_peer_support_group(self):
        if not self.registered_microproviders:
            self.model.logger.info("No registered microproviders to run a peer support group.")
            return
        
        micros_attending = []

        for micro_id in self.registered_microproviders:
            if random.random() < self.model.p_micro_support_attendance:  # Probability check for attendance
                micro = self.model.microprovider_agent_registry[micro_id]['agent_object']
                # Encourage improvement in quality, capped at 1.0
                micro.micro_quality = min(micro.micro_quality + 0.05, 1.0)
                micros_attending.append(micro_id)

        for micro_id in micros_attending:
            micro = self.model.microprovider_agent_registry[micro_id]['agent_object']
            for peer_id in micros_attending:
                if peer_id != micro_id and peer_id not in micro.microprovider_peers:
                    micro.microprovider_peers.append(peer_id)
                    self.model.logger.info(
                        f"Microprovider {micro_id} added Microprovider {peer_id} to their peers list."
                    )

    def run_resident_peer_support_group(self): 
        residents_attending = []

        for resident_id in self.model.resident_agent_registry.keys():
            if random.random() < self.model.p_resident_support_attendance:  # Probability check for attendance
                resident = self.model.resident_agent_registry[resident_id]\
                    ['agent_object']
                residents_attending.append(resident)

        for resident in residents_attending:
            for other_resident in residents_attending:
                if other_resident.unique_id != resident.unique_id:
                    for micro in other_resident.microproviders_to_recommend:
                        if (micro not in resident.microproviders_to_recommend and
                            micro not in resident.blacklisted_microproviders and
                            self.model.microprovider_agent_registry[micro]\
                            ['agent_object'].micro_quality >= resident.micro_quality_threshold):
                            resident.microproviders_to_recommend.append(micro)
                            self.model.logger.info(
                                f"Resident {resident.unique_id} received a recommendation for Microprovider {micro} from Resident {other_resident.unique_id} in a peer support group."
                            )
            if resident.micro_quality_threshold < self.coord_micro_quality_threshold:
                resident.micro_quality_threshold = min(
                    resident.micro_quality_threshold + 0.05,
                    self.coord_micro_quality_threshold)
                self.model.logger.info(
                    f"Resident {resident.unique_id} increased their microprovider quality threshold to {resident.micro_quality_threshold} after attending a peer support group."
                ) 
            if resident.micro_quality_threshold > self.coord_micro_quality_threshold:
                resident.micro_quality_threshold = max(
                    resident.micro_quality_threshold - 0.05,
                    self.coord_micro_quality_threshold)
                self.model.logger.info(
                    f"Resident {resident.unique_id} decreased their microprovider quality threshold to {resident.micro_quality_threshold} after attending a peer support group."
                )  

    def step(self):
        if self.model.step_count % self.model.microprovider_coordinator_group_interval == 0:
            self.run_microprovider_peer_support_group()
        
        if self.model.step_count % self.model.resident_coordinator_group_interval == 0:
            self.run_resident_peer_support_group()

        if random.random() < 0.1:
            self.microprovider_outreach()

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
                 p_resident_leave,
                 p_microprovider_leave,
                 p_microprovider_join,
                 p_use_coordinator,
                 p_approach_random_micro,
                 p_review_care,
                 p_promote_micro,
                 coordinator_has_threshold,
                 coord_micro_quality_threshold,
                 micro_join_coord,
                 resident_coordinator_group_interval,
                 microprovider_coordinator_group_interval,
                 p_micro_support_attendance,
                 p_resident_support_attendance,
                 buffer_steps=200,
                 random_seed=np.random.seed(),
                 annual_population_growth_rate=0.011):
        super().__init__()

        '''setting up model logger'''
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

        self.logger.addHandler(buffer_handler)
        self.logger.propagate = False

        '''setting up model warmup and buffer before main run'''
                # Add a warming-up flag
        self.warming_up = True
        self.warming_up_step = None

        self.buffer_steps = buffer_steps
        self.buffer_active = False
        self.buffer_step_count = 0 
        self.buffer_end_step = None

        # model env variables
        self.running = True
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.step_count = 0
            
        # agent number variables
        self.num_resident_agents = N_RESIDENT_AGENTS
        self.num_microprovider_agents = N_MICROPROVIDER_AGENTS
        '''unpaid carers not used'''
        self.num_unpaidcare_agents = N_UNPAIDCARE_AGENTS
        self.num_coordinator_agents = N_COORDINATOR_AGENTS

        # variables for probabilities and thresholds
        self.resident_care_need_min = resident_care_need_min
        self.resident_care_need_max = resident_care_need_max
        self.microprovider_care_cap_min = microprovider_care_cap_min
        self.microprovider_care_cap_max = microprovider_care_cap_max

        # probabilities for resident behaviours
        self.p_use_coordinator = p_use_coordinator
        self.p_approach_random_micro = p_approach_random_micro
        self.p_review_care = p_review_care
        self.p_promote_micro = p_promote_micro

        # coord attributes
        self.coord_micro_quality_threshold = coord_micro_quality_threshold
        self.coordinator_has_threshold = coordinator_has_threshold

        self.resident_coordinator_group_interval = resident_coordinator_group_interval
        self.microprovider_coordinator_group_interval = microprovider_coordinator_group_interval

        self.p_micro_support_attendance = p_micro_support_attendance
        self.p_resident_support_attendance = p_resident_support_attendance

        # probabilities for microprovider behaviours
        self.micro_join_coord = micro_join_coord

        # controlling resident growth
        self.annual_population_growth_rate = annual_population_growth_rate
        self.p_resident_leave = p_resident_leave
        self.p_microprovider_leave = p_microprovider_leave
        self.p_microprovider_join = p_microprovider_join
        self.annual_population_growth_rate = annual_population_growth_rate
        self.weekly_population_growth_rate = (1 + annual_population_growth_rate)\
        ** (1/52)
        self.fractional_growth_accumulator = 0

        # counters for residents leaving the model - to monitor turnover
        self.residents_leaving = 0

        # rolling count of approaches
        self.num_micros_approached_randomly = 0
        self.num_micros_approached_recommended = 0
        self.num_micros_approached_carer_recommended = 0
        self.num_micros_approached_coordinator = 0
        # step-by-step count of approaches
        self.step_micros_approached_randomly = 0
        self.step_micros_approached_recommended = 0
        self.step_micros_approached_carer_recommended = 0
        self.step_micros_approached_coordinator = 0

        # initialise agent registries
        self.resident_agent_registry = {}
        self.microprovider_agent_registry = {}
        '''unpaid carers not used'''
        self.unpaidcarer_agent_registry = {}
        self.coordinator_agent_registry = {}

        # placing coordinator agent
        for i in range(self.num_coordinator_agents):
            a = Coordinator_Agent(i, self)
            self.schedule.add(a)
            self.coordinator_agent_registry[i] = {
                'agent_object': a,
                'agent_id': i,
                'registered_microproviders': [],
                'coord_micro_quality_threshold': a.coord_micro_quality_threshold
            }

            # Place the coordinator at the center of the grid
            center_x = self.grid.width // 2
            center_y = self.grid.height // 2
            self.grid.place_agent(a, (center_x, center_y))

            # Update the coordinator's position in the registry
            self.coordinator_agent_registry[a.unique_id]['pos'] = a.pos

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

            self.resident_agent_registry[a.unique_id]['pos'] = a.pos

        '''
        microprovider agents are drip fed into the model when warming up, 
        unpaid care agents are not used
        '''
        
        '''setting up data collector'''
        self.datacollector = DataCollector(
            model_reporters={
                "number_of_micros_randomly_approached": self.calc_num_micros_approached_randomly,
                "number_of_micros_recommended": self.calc_num_micros_approached_recommended,
                "number_of_micros_approached_coordinator": self.calc_num_micros_approached_coordinator,
                "step_micros_approached_randomly": lambda m: m.step_micros_approached_randomly,
                "step_micros_approached_recommended": lambda m: m.step_micros_approached_recommended,
                "step_micros_approached_carer_recommended": lambda m: m.step_micros_approached_carer_recommended,
                "step_micros_approached_coordinator": lambda m: m.step_micros_approached_coordinator,
                "calc_is_receiving_care": self.calc_receiving_care,
                "calc_micros": self.calc_num_microproviders,
                "resident_population": lambda m: m.num_resident_agents,
                "avg_packages_of_care": self.calc_avg_packages_of_care,
                "avg_connected_microproviders": self.calc_avg_connected_microproviders,
               # "coordinator_register_size": self.calc_coordinator_register_size
            }
        )

    '''functions to access logs'''
    def print_logs(self):
        """Print all accumulated log messages."""
        self.log_buffer.seek(0)
        print(self.log_buffer.read())
        
    def clear_logs(self):
        """Clear all accumulated log messages."""
        self.log_buffer.truncate(0)
        self.log_buffer.seek(0)

    '''function to manage resident population growth'''
    def increase_residents(self):
        """
        Increase the number of residents based on the weekly population growth rate
        and the number of residents leaving the model. Ensure that residents who
        leave are replaced.
        """
        if self.warming_up or self.buffer_active:
            return

        # Calculate the exact number of new residents to add for net growth
        exact_growth = self.num_resident_agents * self.weekly_population_growth_rate
        net_growth = int(exact_growth) - self.num_resident_agents  # Difference between current and target population

        # Ensure net_growth is non-negative
        if net_growth < 0:
            net_growth = 0

        # Add residents to replace those who left
        net_growth += self.residents_leaving

        # Add new residents based on net growth
        for _ in range(net_growth):
            new_id = max(self.resident_agent_registry.keys(), default=0) + 1
            self._add_new_resident(new_id)
            self.logger.info(f"New resident {new_id} added to replace or grow population.")

        # Handle fractional growth probabilistically
        fractional_growth = exact_growth - int(exact_growth)
        self.fractional_growth_accumulator += fractional_growth

        if self.fractional_growth_accumulator >= 1:
            new_id = max(self.resident_agent_registry.keys(), default=0) + 1
            self._add_new_resident(new_id)
            self.fractional_growth_accumulator -= 1
            self.logger.info(f"New resident {new_id} added due to fractional growth.")

        self.logger.info(f"Residents increased by {net_growth} with fractional growth adjustments.")
        
    '''function to manage warming-up phase and then tranistion to buffer phase'''
    def check_warming_up(self):
        """
        Check if the warming-up condition is met.
        The warming-up period ends when 18% of residents are receiving care.
        """
        if not self.warming_up:
            return

        # Calculate the percentage of residents receiving care
        total_residents = len(self.resident_agent_registry)
        receiving_care = self.calc_receiving_care()
        receiving_care_percentage = receiving_care / total_residents * 100

        # Debugging logs
        self.logger.info(f"Step {self.step_count}: Total Residents = {total_residents}")
        self.logger.info(f"Step {self.step_count}: Residents Receiving Care = {receiving_care}")
        self.logger.info(f"Step {self.step_count}: Receiving Care Percentage = {receiving_care_percentage:.2f}%")

        # Check if the warming-up threshold is reached
        if receiving_care_percentage >= 18:
            if self.warming_up_step is None:
                self.warming_up_step = self.step_count

            self.warming_up = False
            self.logger.info(f"Warming-up period ended on step {self.step_count}")

            self.buffer_active = True
            self.logger.info(f"Buffer phase activated at step {self.step_count}")

    '''function to manage buffer phase after warming-up'''
    def check_buffer_active(self):
        if not self.buffer_active:
            return

        self.buffer_step_count += 1
        self.logger.info(f"Buffer Step Count: {self.buffer_step_count}")

        if self.buffer_step_count >= self.buffer_steps:
            self.buffer_active = False
            self.buffer_end_step = self.buffer_end_step or self.step_count
            self.logger.info(f"Buffer period ended at step {self.step_count}")

            # Reset counters after the buffer period ends
            self._reset_counters()
            self.logger.info("Counters have been reset after the buffer period.")

    '''function to reset counters after buffer period'''
    def _reset_counters(self):
        """Reset counters related to micro approaches."""
        self.num_micros_approached_randomly = 0
        self.num_micros_approached_recommended = 0
        self.num_micros_approached_carer_recommended = 0
        self.num_micros_approached_coordinator = 0

        self.step_micros_approached_randomly = 0
        self.step_micros_approached_recommended = 0
        self.step_micros_approached_carer_recommended = 0
        self.step_micros_approached_coordinator = 0

    """
    Add new micro-providers during warming-up based on the care threshold
    and after warming-up based on a small random chance.
    """ 
    def add_new_agents(self):
        # During warming-up: Add microproviders if care percentage is below the threshold
        if self.warming_up:
            total_residents = len(self.resident_agent_registry)
            receiving_care = self.calc_receiving_care()
            receiving_care_percentage = receiving_care / total_residents * 100

            # Target care percentage during warming-up
            target_care_percentage = 18

            self.logger.info(f"Warming-Up: Care Percentage = {receiving_care_percentage:.2f}%")

            if receiving_care_percentage < target_care_percentage:
                # Add microproviders to meet the care demand
                num_new_microproviders = 1  # Gradual addition (e.g., 1 per step)
                for _ in range(num_new_microproviders):
                    new_id = max(self.microprovider_agent_registry.keys(), default=0) + 1
                    self._add_new_microprovider(new_id)
                    self.logger.info(f"Microprovider {new_id} added during warming-up.")

        # buffer between warming-up and normal operation - do nothing
        if self.buffer_active:
            return
        
        # After warming-up: Add microproviders randomly
        if not self.warming_up and not self.buffer_active and random.random() < self.p_microprovider_join:
            new_id = max(self.microprovider_agent_registry.keys(), default=0) + 1
            self._add_new_microprovider(new_id)
            self.logger.info(f"Microprovider {new_id} joined the model randomly.")
            # print(f"Microprovider {new_id} joined the model randomly.")

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
        
        self.resident_agent_registry[a.unique_id]['pos'] = a.pos

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
            'microprovider_peers': [],
            'packages_of_care_delivered': []
        }
        
        try:
            start_cell = self.grid.find_empty()
            self.grid.place_agent(a, start_cell)
        except:
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))

        self.microprovider_agent_registry[a.unique_id]['pos'] = a.pos

        self.num_microprovider_agents += 1
        self.logger.info(
            f"New Microprovider {new_id} joined with {a.care_capacity}")
     
    '''remove agents from model based on conditions'''
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
        # Prevent agent removal during warmup or buffer phases
        if self.warming_up or self.buffer_active:
            return

        # Reset the residents_leaving counter at the start of each step
        self.residents_leaving = 0

        # Check residents
        for resident_id in list(self.resident_agent_registry.keys()):
            resident = self.resident_agent_registry[resident_id]['agent_object']
            if resident.care_needs_met and random.random() < self.p_resident_leave:
                self.remove_agent(resident)
                self.residents_leaving += 1  # Increment the counter
                self.logger.info(f"Resident {resident_id} has left the model")
                
        # Check microproviders
        for micro_id in list(self.microprovider_agent_registry.keys()):
            micro = self.microprovider_agent_registry[micro_id]['agent_object']
            if not micro.residents and random.random() < self.p_microprovider_leave:
                self.remove_agent(micro)
                self.logger.info(f"Microprovider {micro_id} has left the model")

    '''updating agent registries function for the model step function'''
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
            'microprovider_peers': agent.microprovider_peers,
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
            'unpaidcare_rec': agent.unpaidcare_rec,
        })

    '''unpaid carers not in model currently'''
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
    
    def calc_num_micros_approached_coordinator(self):
        num = self.num_micros_approached_coordinator
        return num
    
    def calc_num_microproviders(self):
        return len(self.microprovider_agent_registry)
    
    def calc_receiving_care(self):
        count_receiving_care = 0
        for agent_id in self.resident_agent_registry:
            if self.resident_agent_registry[agent_id]\
                ['packages_of_care_received'] != []:
                count_receiving_care += 1
        return count_receiving_care

    def calc_avg_packages_of_care(self):
        # Filter residents who have at least one package of care
        residents_with_care = [
            agent for agent in self.resident_agent_registry.values()
            if len(agent['packages_of_care_received']) > 0
        ]

        # Calculate the total number of packages of care
        total_packages = sum(
            len(agent['packages_of_care_received'])
            for agent in residents_with_care
        )

        # Calculate the number of residents with care
        num_residents_with_care = len(residents_with_care)

        # Return the average, or 0 if no residents have care
        return total_packages / num_residents_with_care if num_residents_with_care > 0 else 0

    def calc_avg_connected_microproviders(self):
        total_connections = sum(
            len(agent['allocated_microproviders'])
            for agent in self.resident_agent_registry.values()
        )
        num_residents = len(self.resident_agent_registry)
        return total_connections / num_residents if num_residents > 0 else 0

    # model step
    def step(self):
        self.logger.info(f"Model step {self.step_count}")

        # Check if the warming-up condition is met
        self.check_warming_up()
        # print(f"Warming up status: {self.warming_up}")

        self.check_buffer_active()
        # print(f"Buffer active status: {self.buffer_active}")

        # Add new agents during warming-up or randomly after warming-up
        self.add_new_agents()

        # Check agent removals and increase residents after warming-up
        self.check_agent_removals()
        # print(f"Agent removals checked at step {self.step_count}")
        self.increase_residents()
        # print(f"Residents increased at step {self.step_count}")

        # Update registries and step each agent
        for agent in self.schedule.agents:
            self._update_agent_registry(agent)
            agent.step()
            # print(f"Stepped agent {agent.unique_id} of type {type(agent).__name__}")

        self.datacollector.collect(self)

        # Reset step-by-step counters
        self.step_micros_approached_randomly = 0
        self.step_micros_approached_recommended = 0
        self.step_micros_approached_carer_recommended = 0
        self.step_micros_approached_coordinator = 0

        self.schedule.step()
        self.step_count += 1

"""end of model"""

def run_care_model(params=None):
    """
    Run the Care_Model simulation using parameters from a dictionary.

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

    # Extract the number of years from the params dictionary (default: 5 years)
    num_years = params.get("num_years", 5)

    # Initialize the model
    model = Care_Model(
        N_RESIDENT_AGENTS=params.get("n_residents", 833),
        N_MICROPROVIDER_AGENTS=params.get("n_microproviders", 0),
        N_UNPAIDCARE_AGENTS=params.get("n_unpaidcarers", 0),
        N_COORDINATOR_AGENTS=params.get("n_coordinators", 0),
        # grid and misc
        width=params.get("width", 50),
        height=params.get("height", 50),
        random_seed=params.get("random_seed", 42),
        # Care needs and capacities
        resident_care_need_min=params.get("resident_care_need_min", 1),
        resident_care_need_max=params.get("resident_care_need_max", 20),
        microprovider_care_cap_min=params.get("microprovider_care_cap_min", 5),
        microprovider_care_cap_max=params.get("microprovider_care_cap_max", 40),
        # Probabilities and thresholds
        p_resident_leave=params.get("p_resident_leave", 0.001),
        p_microprovider_leave=params.get("p_microprovider_leave", 0.001),
        p_use_coordinator=params.get("p_use_coordinator", 0.01),
        p_approach_random_micro=params.get("p_approach_random_micro", 0.01),
        p_review_care=params.get("p_review_care", 0.001),
        p_promote_micro=params.get("p_promote_micro", 0.001),
        p_microprovider_join=params.get("p_microprovider_join", 0.01),  # Updated parameter name
        # microprovider attributes
        micro_join_coord=params.get("micro_join_coord", 0.5),
        #coordinator attributes
        coord_micro_quality_threshold=params.get("coord_micro_quality_threshold", 0.5),
        coordinator_has_threshold=params.get("coordinator_has_threshold", True),
        resident_coordinator_group_interval=params.get("resident_coordinator_group_interval", 4),
        microprovider_coordinator_group_interval=params.get("microprovider_coordinator_group_interval", 4),
        p_micro_support_attendance=params.get("p_micro_support_attendance", 0.1),
        p_resident_support_attendance=params.get("p_resident_support_attendance", 0.01)
        )

    # Run the model until the warming-up phase ends
    while model.warming_up:
        model.step()
        # print(f"Warming up... Step {model.step_count}")

    while model.buffer_active:
        model.step()
        # print(f"Buffering... Step {model.step_count}")

    # Calculate the total number of steps to run after warming-up
    total_steps = num_years * 52  # 1 year = 52 weeks

    # Run the model for the specified number of steps
    for _ in range(total_steps):
        model.step()
        # print(f"Running... Step {model.step_count}")

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

    # Export registries to CSV files
    data_resident_registry.to_csv("resident_registry.csv", index=False)
    data_microprovider_registry.to_csv("microprovider_registry.csv", index=False)
    if not data_coord_registry.empty:
        data_coord_registry.to_csv("coordinator_registry.csv", index=False)

    # Return all results as a dictionary
    return {
        "model": model,
        "data": data,
        "data_resident_registry": data_resident_registry,
        "data_microprovider_registry": data_microprovider_registry,
        "data_coord_registry": data_coord_registry,
    }

# results = run_care_model(params={"num_years": 5, "n_coordinators":1})
# print(results['model'].num_resident_agents)
# print(len(results['data_resident_registry']))