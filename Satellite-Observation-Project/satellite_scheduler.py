import pulp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
import json
import os
from time import time

class Target:
    """Represents an observation target - στόχος παρατήρησης"""
    def __init__(self, id: int, name: str, lat: float, lon: float, 
                 priority: float, min_elevation: float = 30.0):
        self.id = id
        self.name = name
        self.lat = lat # latitude δηλαδ΄ή Γεωγραφικό πλάτος -90° (Νότιος Πόλος) έως +90° (Βόρειος Πόλος)
        self.lon = lon # longitude δηλαδή Γεωγραφικό μήκος -180° (Δυτικό Πόλο) έως +180° (Ανατολικό Πόλο)
        self.priority = priority
        self.min_elevation = min_elevation # ελάχιστη γωνία ύψους για παρατήρηση (σε μοίρες) - συνήθως 30° ή 40°

class Satellite:
    """Represents a satellite with observation capabilities - δορυφόρος με δυνατότητες παρατήρησης"""
    def __init__(self, id: int, name: str, memory_capacity: float, 
                 power_capacity: float, data_rate: float, setup_time: float = 2.0):
        self.id = id
        self.name = name
        self.memory_capacity = memory_capacity  # GB
        self.power_capacity = power_capacity    # Watts
        self.data_rate = data_rate              # Mbps
        self.setup_time = setup_time            # minutes (NEW)

class Observation:
    """Represents a potential observation opportunity - υποψήφια παρατήρηση κάποιου στόχου"""
    def __init__(self, target_id: int, satellite_id: int, start_time: datetime,
                 end_time: datetime, elevation: float, data_volume: float,
                 power_required: float):
        self.target_id = target_id
        self.satellite_id = satellite_id
        self.start_time = start_time # έναρξη παρατήρησης
        self.end_time = end_time
        self.duration = (end_time - start_time).total_seconds() / 60  # minutes
        self.elevation = elevation
        self.data_volume = data_volume  # GB
        self.power_required = power_required  # Watts

class SatelliteScheduler:
    """Enhanced MILP-based satellite observation scheduler with setup time and conflict degree
    handling - Βελτιωμένος προγραμματιστής παρατήρησης δορυφόρων με MILP με setup time και conflict degree"""
    
    def __init__(self, satellites: List[Satellite], targets: List[Target], 
                 time_horizon: int = 24, use_conflict_degree: bool = True):
        self.satellites = satellites
        self.targets = targets
        self.time_horizon = time_horizon  # hours
        self.use_conflict_degree = use_conflict_degree  # NEW
        self.observations = []
        self.conflict_degrees = {}  # NEW: stores conflict degree for each observation
        self.model = None
        self.solution = None
        
    def generate_observation_opportunities(self, start_time: datetime, 
                                         time_step: int = 10) -> List[Observation]:
        """Generate synthetic observation opportunities"""
        observations = []
        obs_id = 0
        
        for satellite in self.satellites: # για καθέ δορυφόρο
            for target in self.targets: # για κάθε στόχο
                # Generate multiple opportunities per target-satellite pair
                for hour in range(0, self.time_horizon, 2): # κάθε 2 ώρες
                    # Simulate orbital mechanics with some randomness - δορυφόρος σε τροχιά
                    # Assume visibility every 2 hours with some random variation
                    if random.random() < 0.3:  # 30% chance of visibility
                        start = start_time + timedelta(hours=hour, 
                                                        minutes=random.randint(0, 119))
                        duration = random.randint(5, 15)  # 5-15 minutes
                        end = start + timedelta(minutes=duration)
                        elevation = random.uniform(target.min_elevation, 85)
                        
                        # Data volume based on observation duration and satellite capability
                        data_volume = (duration / 60) * satellite.data_rate * 0.125 / 1000  # GB
                        power_required = random.uniform(50, 150)  # Watts
                        
                        obs = Observation(
                            target_id=target.id,
                            satellite_id=satellite.id,
                            start_time=start,
                            end_time=end,
                            elevation=elevation,
                            data_volume=data_volume,
                            power_required=power_required
                        )
                        observations.append(obs)
                        obs_id += 1
        
        self.observations = observations
        # Calculate conflict degrees after generating observations
        if self.use_conflict_degree:
            self._calculate_conflict_degrees()
        return observations
    
    def check_temporal_conflict_with_setup(self, obs1: Observation, obs2: Observation) -> bool:
        """Check if two observations conflict in time considering setup time"""
        if obs1.satellite_id != obs2.satellite_id:
            return False
        
        satellite = next(s for s in self.satellites if s.id == obs1.satellite_id)
        setup_time_delta = timedelta(minutes=satellite.setup_time)
        
        # Check if there's enough gap for setup time between observations
        if obs1.end_time <= obs2.start_time:
            # obs1 ends before obs2 starts - check if gap is sufficient 
            # η παρατήρηση obs1 τελειώνει πριν την obs2 ξεκινήσει => έλεγχος αν το κενό είναι αρκετό 	
            return (obs2.start_time - obs1.end_time) < setup_time_delta
        elif obs2.end_time <= obs1.start_time:
            # obs2 ends before obs1 starts - check if gap is sufficient
            return (obs1.start_time - obs2.end_time) < setup_time_delta
        else:
            # Direct temporal overlap
            return True
    
    def check_temporal_conflict(self, obs1: Observation, obs2: Observation) -> bool:
        """Original conflict check method - without setup time
        Check if two observations conflict in time Έκδοση που δεν λαμβάνει υπόψιν της το set up time"""
        if obs1.satellite_id != obs2.satellite_id:
            return False
        
        return not (obs1.end_time <= obs2.start_time or obs2.end_time <= obs1.start_time)
    
    def _calculate_conflict_degrees(self):
        """Calculate conflict degree for each observation"""
        self.conflict_degrees = {}
        
        for i, obs in enumerate(self.observations):
            conflict_count = 0
            
            for j, other_obs in enumerate(self.observations):
                if i != j and self.check_temporal_conflict_with_setup(obs, other_obs):
                    conflict_count += 1
            
            self.conflict_degrees[i] = conflict_count
        
        print(f"Calculated conflict degrees - Max: {max(self.conflict_degrees.values())}, "
              f"Min: {min(self.conflict_degrees.values())}, "
              f"Avg: {np.mean(list(self.conflict_degrees.values())):.2f}")
    
    def build_milp_model(self):
        """Build the Mixed Integer Linear Programming model with enhanced constraints"""
        
        # Create the optimization problem
        self.model = pulp.LpProblem("SatelliteScheduling", pulp.LpMaximize)
        
        # Decision variables: x_ij = 1 if observation i is selected, 0 otherwise
        x = {}
        for i, obs in enumerate(self.observations):
            x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')
        
        # Enhanced Objective function with conflict degree weighting
        objective = 0
        for i, obs in enumerate(self.observations):
            target = next(t for t in self.targets if t.id == obs.target_id)
            
            # Base weight calculation
            base_weight = target.priority * (obs.elevation / 90.0) * (1.0 / (obs.data_volume + 0.1))
            
            # Apply conflict degree weighting if enabled
            if self.use_conflict_degree and i in self.conflict_degrees:
                conflict_penalty = 1 + self.conflict_degrees[i] * 0.1  # Adjustable penalty factor
                weight = base_weight / conflict_penalty
            else:
                weight = base_weight
            
            objective += weight * x[i]
        
        self.model += objective, "Total_Weighted_Value"
        
        # Enhanced Constraint 1: No temporal conflicts with setup time consideration
        conflict_pairs = []
        for i in range(len(self.observations)):
            for j in range(i + 1, len(self.observations)):
                if self.check_temporal_conflict_with_setup(self.observations[i], self.observations[j]):
                    conflict_pairs.append((i, j))
                    self.model += x[i] + x[j] <= 1, f"Conflict_Setup_{i}_{j}"
        
        # Constraint 2: Memory capacity constraints per satellite
        for satellite in self.satellites:
            satellite_obs = [i for i, obs in enumerate(self.observations) 
                           if obs.satellite_id == satellite.id]
            if satellite_obs:
                memory_constraint = pulp.lpSum([
                    self.observations[i].data_volume * x[i] for i in satellite_obs
                ])
                self.model += memory_constraint <= satellite.memory_capacity, \
                             f"Memory_Satellite_{satellite.id}"
        
        # Constraint 3: Power capacity constraints per satellite per time window
        time_windows = self._create_time_windows()
        for satellite in self.satellites:
            for tw_start, tw_end in time_windows:
                satellite_obs_in_window = [
                    i for i, obs in enumerate(self.observations)
                    if (obs.satellite_id == satellite.id and 
                        not (obs.end_time <= tw_start or obs.start_time >= tw_end))
                ]
                if satellite_obs_in_window:
                    power_constraint = pulp.lpSum([
                        self.observations[i].power_required * x[i] 
                        for i in satellite_obs_in_window
                    ])
                    self.model += power_constraint <= satellite.power_capacity, \
                                 f"Power_Satellite_{satellite.id}_Window_{tw_start.hour}"
        
        # Constraint 4: At most one observation per target (optional)
        for target in self.targets:
            target_obs = [i for i, obs in enumerate(self.observations) 
                         if obs.target_id == target.id]
            if len(target_obs) > 1:
                self.model += pulp.lpSum([x[i] for i in target_obs]) <= 1, \
                             f"OneObsPerTarget_{target.id}"
        
        # NEW Constraint 5: Setup time sequencing constraints (alternative formulation)
        # This adds additional constraints to ensure proper sequencing with setup times
        for satellite in self.satellites:
            satellite_obs_indices = [i for i, obs in enumerate(self.observations) 
                                   if obs.satellite_id == satellite.id]
            
            # For each pair of observations on the same satellite
            for i in satellite_obs_indices:
                for j in satellite_obs_indices:
                    if i != j:
                        obs_i = self.observations[i]
                        obs_j = self.observations[j]
                        
                        # If obs_i could potentially be followed by obs_j
                        time_gap = (obs_j.start_time - obs_i.end_time).total_seconds() / 60
                        if 0 < time_gap < satellite.setup_time:
                            # They're too close - can't both be selected
                            self.model += x[i] + x[j] <= 1, f"Setup_Gap_{i}_{j}"
        
        self.decision_vars = x
        print(f"Enhanced model built with {len(self.observations)} potential observations")
        print(f"Found {len(conflict_pairs)} temporal conflicts (including setup time)")
        if self.use_conflict_degree:
            print(f"Conflict degree weighting: ENABLED")
        else:
            print(f"Conflict degree weighting: DISABLED")
        
    def _create_time_windows(self, window_size_hours: int = 4) -> List[Tuple[datetime, datetime]]:
        """Create time windows for power constraint analysis"""
        if not self.observations:
            return []
            
        base_time = self.observations[0].start_time.replace(hour=0, minute=0, second=0)
        windows = []
        
        for i in range(0, self.time_horizon, window_size_hours):
            start = base_time + timedelta(hours=i)
            end = base_time + timedelta(hours=i + window_size_hours)
            windows.append((start, end))
        
        return windows
    
    def solve(self, solver=None, time_limit=300):
        """Solve the MILP model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_milp_model() first.")
        
        # Use CBC as default solver
        if solver is None:
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)

        print("Solving enhanced MILP model...")
        self.model.solve(solver)
        
        # Check solution status
        status = pulp.LpStatus[self.model.status]
        print(f"Solution status: {status}")
        
        if status == 'Optimal':
            self.solution = {
                'status': status,
                'objective_value': pulp.value(self.model.objective),
                'selected_observations': []
            }
            
            for i, var in self.decision_vars.items():
                if pulp.value(var) == 1:
                    self.solution['selected_observations'].append(i)
            
            print(f"Optimal objective value: {self.solution['objective_value']:.3f}")
            print(f"Selected {len(self.solution['selected_observations'])} observations")
            
        return self.solution
    
    def analyze_solution(self):
        """Analyze and display solution statistics with setup time considerations"""
        if not self.solution or self.solution['status'] != 'Optimal':
            print("No optimal solution to analyze")
            return
        
        selected_obs_idx = self.solution['selected_observations']
        selected_observations = [self.observations[i] for i in selected_obs_idx]
        
        # Statistics by satellite
        sat_stats = {}
        for satellite in self.satellites:
            sat_obs = [obs for obs in selected_observations if obs.satellite_id == satellite.id]
            total_data = sum(obs.data_volume for obs in sat_obs)
            total_time = sum(obs.duration for obs in sat_obs)
            
            # Calculate total time including setup times
            sat_obs_sorted = sorted(sat_obs, key=lambda x: x.start_time)
            total_time_with_setup = total_time
            if len(sat_obs_sorted) > 1:
                total_time_with_setup += (len(sat_obs_sorted) - 1) * satellite.setup_time
            
            sat_stats[satellite.name] = {
                'observations': len(sat_obs),
                'total_data_gb': total_data,
                'total_time_min': total_time,
                'total_time_with_setup_min': total_time_with_setup,
                'setup_time_overhead_min': total_time_with_setup - total_time,
                'memory_utilization': (total_data / satellite.memory_capacity) * 100
            }
        
        # Statistics by target priority
        target_coverage = {}
        for target in self.targets:
            target_obs = [obs for obs in selected_observations if obs.target_id == target.id]
            target_coverage[target.name] = {
                'covered': len(target_obs) > 0,
                'priority': target.priority,
                'observations': len(target_obs)
            }
        
        # Conflict degree statistics
        conflict_stats = {}
        if self.use_conflict_degree:
            selected_conflict_degrees = [self.conflict_degrees[i] for i in selected_obs_idx]
            conflict_stats = {
                'avg_conflict_degree': np.mean(selected_conflict_degrees),
                'max_conflict_degree': max(selected_conflict_degrees),
                'min_conflict_degree': min(selected_conflict_degrees)
            }
        
        return {
            'satellite_stats': sat_stats,
            'target_coverage': target_coverage,
            'conflict_stats': conflict_stats,
            'total_observations': len(selected_observations),
            'total_data_volume': sum(obs.data_volume for obs in selected_observations)
        }
    
    def validate_schedule_constraints(self):
        """Validate that the solution respects setup time constraints"""
        if not self.solution or self.solution['status'] != 'Optimal':
            print("No solution to validate")
            return False
        
        selected_obs_idx = self.solution['selected_observations']
        selected_observations = [self.observations[i] for i in selected_obs_idx]
        
        violations = []
        
        for satellite in self.satellites:
            sat_obs = [obs for obs in selected_observations if obs.satellite_id == satellite.id]
            sat_obs_sorted = sorted(sat_obs, key=lambda x: x.start_time)
            
            for i in range(len(sat_obs_sorted) - 1):
                obs1 = sat_obs_sorted[i]
                obs2 = sat_obs_sorted[i + 1]
                
                time_gap = (obs2.start_time - obs1.end_time).total_seconds() / 60
                if time_gap < satellite.setup_time:
                    violations.append({
                        'satellite': satellite.name,
                        'obs1_end': obs1.end_time,
                        'obs2_start': obs2.start_time,
                        'gap_minutes': time_gap,
                        'required_setup_minutes': satellite.setup_time
                    })
        
        if violations:
            print(f"Found {len(violations)} setup time violations!")
            for v in violations:
                print(f"  {v['satellite']}: Gap {v['gap_minutes']:.1f}min < Required {v['required_setup_minutes']:.1f}min")
            return False
        else:
            print("Schedule validation: All setup time constraints satisfied ✓")
            return True
    
    def visualize_schedule(self, figname="enhanced_satellite_schedule.png",figsize=(15, 12)):
        """Create an enhanced Gantt chart visualization of the schedule"""
        if not self.solution or self.solution['status'] != 'Optimal':
            print("No optimal solution to visualize")
            return
        
        selected_obs_idx = self.solution['selected_observations']
        selected_observations = [self.observations[i] for i in selected_obs_idx]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[4, 2, 1])
        
        # Gantt chart with setup times
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.satellites)))
        satellite_colors = {sat.id: colors[i] for i, sat in enumerate(self.satellites)}
        
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        for satellite in self.satellites:
            sat_obs = [obs for obs in selected_observations if obs.satellite_id == satellite.id]
            sat_obs_sorted = sorted(sat_obs, key=lambda x: x.start_time)
            
            for i, obs in enumerate(sat_obs_sorted):
                start_hours = (obs.start_time - selected_observations[0].start_time).total_seconds() / 3600
                duration_hours = obs.duration / 60
                
                target = next(t for t in self.targets if t.id == obs.target_id)
                
                # Draw observation
                ax1.barh(y_pos, duration_hours, left=start_hours, 
                        color=satellite_colors[satellite.id], alpha=0.7,
                        label=f"{satellite.name}" if i == 0 else "")
                
                # Add target name annotation
                ax1.text(start_hours + duration_hours/2, y_pos, target.name, 
                        ha='center', va='center', fontsize=8, rotation=45)
                
                # Draw setup time if not the first observation
                if i > 0:
                    setup_start = start_hours - (satellite.setup_time / 60)
                    setup_duration = satellite.setup_time / 60
                    ax1.barh(y_pos, setup_duration, left=setup_start,
                            color='red', alpha=0.3, hatch='///')
            
            y_labels.append(satellite.name)
            y_ticks.append(y_pos)
            y_pos += 1
        
        ax1.set_ylabel('Satellites')
        ax1.set_xlabel('Time (hours)')
        ax1.set_title('Enhanced Satellite Observation Schedule (Red = Setup Time)')
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_labels)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Resource utilization chart
        satellites_names = [sat.name for sat in self.satellites]
        memory_util = []
        setup_overhead = []
        
        for satellite in self.satellites:
            sat_obs = [obs for obs in selected_observations if obs.satellite_id == satellite.id]
            total_data = sum(obs.data_volume for obs in sat_obs)
            util = (total_data / satellite.memory_capacity) * 100
            memory_util.append(util)
            
            # Calculate setup time overhead
            if len(sat_obs) > 1:
                overhead = (len(sat_obs) - 1) * satellite.setup_time
            else:
                overhead = 0
            setup_overhead.append(overhead)
        
        bars = ax2.bar(satellites_names, memory_util, 
                      color=[satellite_colors[sat.id] for sat in self.satellites])
        ax2.set_ylabel('Memory Utilization (%)')
        ax2.set_title('Satellite Memory Utilization')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, util in zip(bars, memory_util):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{util:.1f}%', ha='center', va='bottom')
        
        # Setup time overhead chart
        bars3 = ax3.bar(satellites_names, setup_overhead,
                       color=[satellite_colors[sat.id] for sat in self.satellites], alpha=0.6)
        ax3.set_ylabel('Setup Time (min)')
        ax3.set_title('Setup Time Overhead per Satellite')
        
        # Add value labels
        for bar, overhead in zip(bars3, setup_overhead):
            if overhead > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{overhead:.0f}min', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the visualization
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        plot_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_path = os.path.join(plot_dir, figname)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced schedule visualization saved as '{plot_path}'")
        
        return fig


########## create scenarios


# example scenario
def create_example_scenario():
    """Create an example scenario for testing with setup times"""
    
    # Create satellites with different setup times
    satellites = [
        Satellite(1, "Sentinel-1A", memory_capacity=10.0, power_capacity=200, data_rate=100, setup_time=2.0),
        Satellite(2, "Landsat-9", memory_capacity=15.0, power_capacity=180, data_rate=150, setup_time=3.0),
        Satellite(3, "WorldView-3", memory_capacity=8.0, power_capacity=160, data_rate=120, setup_time=1.5)
    ]
    
    # Create targets
    targets = [
        Target(1, "Forest_Fire_California", 37.7749, -122.4194, priority=0.9),
        Target(2, "Crop_Monitoring_Iowa", 41.5868, -93.6250, priority=0.6),
        Target(3, "Urban_Growth_Tokyo", 35.6762, 139.6503, priority=0.7),
        Target(4, "Flood_Assessment_Bangladesh", 23.6850, 90.3563, priority=0.95),
        Target(5, "Ice_Monitoring_Arctic", 71.0, -8.0, priority=0.8),
        Target(6, "Desert_Expansion_Sahara", 23.0, 8.0, priority=0.5),
        Target(7, "Volcano_Activity_Italy", 40.8518, 14.2681, priority=0.85)
    ]
    
    return satellites, targets


####### Run scenarios from JSON file
def run_scenarios_from_json(json_path: str, output_dir: str = "results"):
    """
    Load multiple scenarios from a JSON file, solve each one, and write results back to disk.

    Results:
      └─ <output_dir>/
           ├─ small_result.json
           ├─ medium_result.json
           ├─ large_result.json
           └─ all_results.json   (aggregated)
    """
    
    with open(json_path, "r") as f:
        payload = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for scenario in payload["scenarios"]:
        # --- build objects --------------------------------------------------
        satellites = [Satellite(**s) for s in scenario["satellites"]]
        targets    = [Target(**t)    for t in scenario["targets"]]

        scheduler = SatelliteScheduler(
            satellites,
            targets,
            time_horizon      = scenario.get("time_horizon", 24),
            use_conflict_degree = scenario.get("use_conflict_degree", True),
        )

        # --- end-to-end solve ----------------------------------------------
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        scheduler.generate_observation_opportunities(start)
        scheduler.build_milp_model()
        sol = scheduler.solve()

        analysis = (
            scheduler.analyze_solution()
            if sol and sol["status"] == "Optimal" else None
        )

        # --- persist --------------------------------------------------------
        one_result = {"solution": sol, "analysis": analysis}
        fname = os.path.join(output_dir, f"{scenario['name']}_result.json")
        with open(fname, "w") as fh:
            json.dump(one_result, fh, default=str, indent=2)

        summary[scenario["name"]] = one_result
        print(f"[✓] {scenario['name']} → {fname}")

    # master file
    with open(os.path.join(output_dir, "all_results.json"), "w") as fh:
        json.dump(summary, fh, default=str, indent=2)

    return summary

def extract_scenario_objects(json_path: str, scenario_name: str):
    """
    Load a scenario by name from a JSON file and return (satellites, targets).
    """
    with open(json_path, "r") as f:
        payload = json.load(f)

    # Find the scenario by name
    scenario = next((s for s in payload["scenarios"] if s["name"] == scenario_name), None)
    if not scenario:
        raise ValueError(f"Scenario '{scenario_name}' not found in {json_path}")

    satellites = [Satellite(**s) for s in scenario["satellites"]]
    targets = [Target(**t) for t in scenario["targets"]]
    
    return satellites, targets

########### FLEXIBLE GENERATION 

def generate_scenario(num_sats: int, num_targets: int, name: str = "custom", time_horizon: int = 24):
    """
    Generate a synthetic satellite-target scenario.

    Args:
        num_sats (int): Number of satellites
        num_targets (int): Number of targets
        name (str): Scenario name
        time_horizon (int): Scheduling time window (default: 24)

    Returns:
        dict: JSON-ready scenario dictionary
    """
    scenario = {
        "name": name,
        "time_horizon": time_horizon,
        "use_conflict_degree": True,
        "satellites": [],
        "targets": []
    }

    for i in range(1, num_sats + 1):
        scenario["satellites"].append({
            "id": i,
            "name": f"Sat-{i}",
            "memory_capacity": round(random.uniform(8, 20), 1),
            "power_capacity": random.randint(150, 250),
            "data_rate": random.randint(80, 180),
            "setup_time": round(random.uniform(1.0, 4.0), 1)
        })

    for j in range(1, num_targets + 1):
        scenario["targets"].append({
            "id": j,
            "name": f"Target_{j}",
            "lat": round(random.uniform(-90, 90), 4),
            "lon": round(random.uniform(-180, 180), 4),
            "priority": round(random.uniform(0.5, 1.0), 2)
        })

    return scenario


### Scenario Picker Function => returns satellites, targets, and scenario name from user input
# Η συνάρτηση αυτή επιτρέπει στον χρήστη να επιλέξει ένα σενάριο για εκτέλεση 
def scenario_picker(x :int):
    if x == "1":
        satellites, targets = create_example_scenario()
        return satellites, targets, "example_scenario"
    elif x == "2":
        cur_dir= os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(cur_dir, "data","scenarios.json")
        size_in= input("Chose scenario size: \n1. small\n2. medium\n3. large\n")
        if size_in not in ["small", "medium", "large"]:
            print("Invalid size choice. Exiting.") 
            exit(1)
        satellites, targets = extract_scenario_objects(json_path, f"{size_in}")
        return satellites, targets, size_in
    elif x == "3":
        size_in = input("Enter scenario name (e.g., 'custom_scenario') or leave empty for default: ")
        if not size_in:
            size_in = "custom_scenario"
        num_sats = int(input("Enter number of satellites (e.g., 3): "))
        num_targets = int(input("Enter number of targets (e.g., 5): "))
        time_horizon = int(input("Enter scheduling time horizon in hours (default 24): ") or 24)
        
        scenario = generate_scenario(num_sats, num_targets, name=size_in, time_horizon=time_horizon)
        satellites = [Satellite(**s) for s in scenario["satellites"]]
        targets = [Target(**t) for t in scenario["targets"]]
        
        return satellites, targets, "custom"+size_in
    else:
        print("Invalid choice. Exiting.")
        #exit 
        exit(1)
        return; 

def main():
    """Main execution function with enhanced features"""
    print("=== Enhanced Satellite Observation Scheduling using MILP ===")
    print("Features: Setup Time Constraints + Conflict Degree Weighting\n")
    
    # Create scenario
    x=input("Chose mode: \n1. Run example scenario\n2. Run scenarios from JSON file PREMADE\n3. Generate custom scenario\n")
    satellites, targets, scenarioname = scenario_picker(x)

    print(f"Created scenario with {len(satellites)} satellites and {len(targets)} targets")
    for sat in satellites:
        print(f"  {sat.name}: Setup time = {sat.setup_time} minutes")
    
    # Initialize enhanced scheduler
    scheduler = SatelliteScheduler(satellites, targets, time_horizon=24, use_conflict_degree=True)
    
    # Generate observation opportunities
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    observations = scheduler.generate_observation_opportunities(start_time)
    print(f"Generated {len(observations)} observation opportunities")
    
    # Build and solve enhanced MILP model
    scheduler.build_milp_model()
    start_timer = time()
    solution = scheduler.solve()
    end_timer = time()
    elapsed_sec = end_timer - start_timer
    print(f"\n[**TIMER**] MILP solve time: {elapsed_sec:.2f} seconds")
    if solution and solution['status'] == 'Optimal':
        # Validate constraints
        scheduler.validate_schedule_constraints()
        
        # Analyze solution
        analysis = scheduler.analyze_solution()
        
        print("\n=== Enhanced Solution Analysis ===")
        print(f"Total observations scheduled: {analysis['total_observations']}")
        print(f"Total data volume: {analysis['total_data_volume']:.2f} GB")
        
        if analysis['conflict_stats']:
            print(f"\nConflict Degree Statistics:")
            print(f"  Average conflict degree: {analysis['conflict_stats']['avg_conflict_degree']:.2f}")
            print(f"  Max conflict degree: {analysis['conflict_stats']['max_conflict_degree']}")
            print(f"  Min conflict degree: {analysis['conflict_stats']['min_conflict_degree']}")
        
        print("\nEnhanced Satellite Statistics:")
        for sat_name, stats in analysis['satellite_stats'].items():
            print(f"  {sat_name}:")
            print(f"    Observations: {stats['observations']}")
            print(f"    Data volume: {stats['total_data_gb']:.2f} GB")
            print(f"    Observation time: {stats['total_time_min']:.1f} minutes")
            print(f"    Time with setup: {stats['total_time_with_setup_min']:.1f} minutes")
            print(f"    Setup overhead: {stats['setup_time_overhead_min']:.1f} minutes")
            print(f"    Memory utilization: {stats['memory_utilization']:.1f}%")
        
        print("\nTarget Coverage:")
        for target_name, coverage in analysis['target_coverage'].items():
            status = "✓" if coverage['covered'] else "✗"
            print(f"  {status} {target_name} (Priority: {coverage['priority']:.1f})")
        
        # Visualize results
        scheduler.visualize_schedule(figname=f"satellite_schedule_{scenarioname}.png")
        
    
    return scheduler

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    scheduler = main()