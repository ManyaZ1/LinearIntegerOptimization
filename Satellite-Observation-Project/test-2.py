import pulp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
import json

class Target:
    """Represents an observation target"""
    def __init__(self, id: int, name: str, lat: float, lon: float, 
                 priority: float, min_elevation: float = 30.0):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.priority = priority
        self.min_elevation = min_elevation

class Satellite:
    """Represents a satellite with observation capabilities"""
    def __init__(self, id: int, name: str, memory_capacity: float, 
                 power_capacity: float, data_rate: float):
        self.id = id
        self.name = name
        self.memory_capacity = memory_capacity  # GB
        self.power_capacity = power_capacity    # Watts
        self.data_rate = data_rate              # Mbps

class Observation:
    """Represents a potential observation opportunity"""
    def __init__(self, target_id: int, satellite_id: int, start_time: datetime,
                 end_time: datetime, elevation: float, data_volume: float,
                 power_required: float):
        self.target_id = target_id
        self.satellite_id = satellite_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = (end_time - start_time).total_seconds() / 60  # minutes
        self.elevation = elevation
        self.data_volume = data_volume  # GB
        self.power_required = power_required  # Watts

class SatelliteScheduler:
    """Main MILP-based satellite observation scheduler"""
    
    def __init__(self, satellites: List[Satellite], targets: List[Target], 
                 time_horizon: int = 24):
        self.satellites = satellites
        self.targets = targets
        self.time_horizon = time_horizon  # hours
        self.observations = []
        self.model = None
        self.solution = None
        
    def generate_observation_opportunities(self, start_time: datetime, 
                                         time_step: int = 10) -> List[Observation]:
        """Generate synthetic observation opportunities"""
        observations = []
        obs_id = 0
        
        for satellite in self.satellites:
            for target in self.targets:
                # Generate multiple opportunities per target-satellite pair
                for hour in range(0, self.time_horizon, 2):
                    # Simulate orbital mechanics with some randomness
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
        return observations
    
    def check_temporal_conflict(self, obs1: Observation, obs2: Observation) -> bool:
        """Check if two observations conflict in time for the same satellite"""
        if obs1.satellite_id != obs2.satellite_id:
            return False
        
        return not (obs1.end_time <= obs2.start_time or obs2.end_time <= obs1.start_time)
    
    def build_milp_model(self):
        """Build the Mixed Integer Linear Programming model"""
        
        # Create the optimization problem
        self.model = pulp.LpProblem("SatelliteScheduling", pulp.LpMaximize)
        
        # Decision variables: x_ij = 1 if observation i is selected, 0 otherwise
        x = {}
        for i, obs in enumerate(self.observations):
            x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')
        
        # Objective function: Maximize weighted sum of selected observations
        objective = 0
        for i, obs in enumerate(self.observations):
            target = next(t for t in self.targets if t.id == obs.target_id)
            # Weight by priority, elevation, and inverse of data volume (efficiency)
            weight = target.priority * (obs.elevation / 90.0) * (1.0 / (obs.data_volume + 0.1))
            objective += weight * x[i]
        
        self.model += objective, "Total_Weighted_Value"
        
        # Constraint 1: No temporal conflicts for same satellite
        conflict_pairs = []
        for i in range(len(self.observations)):
            for j in range(i + 1, len(self.observations)):
                if self.check_temporal_conflict(self.observations[i], self.observations[j]):
                    conflict_pairs.append((i, j))
                    self.model += x[i] + x[j] <= 1, f"Conflict_{i}_{j}"
        
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
        
        self.decision_vars = x
        print(f"Model built with {len(self.observations)} potential observations")
        print(f"Found {len(conflict_pairs)} temporal conflicts")
        
    def _create_time_windows(self, window_size_hours: int = 4) -> List[Tuple[datetime, datetime]]:
        """Create time windows for power constraint analysis"""
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
        
        # Use GLPK as default solver
        if solver is None:
            #solver = pulp.GLPK_CMD(path=r'C:\Users\USER\Documents\GitHub\LinearIntegerOptimization\Satellite-Observation-Project\glpk-4.65\w64\glpsol.exe',msg=1, timeLimit=time_limit)
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300)
            self.model.solve(solver)

        print("Solving MILP model...")
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
        """Analyze and display solution statistics"""
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
            
            sat_stats[satellite.name] = {
                'observations': len(sat_obs),
                'total_data_gb': total_data,
                'total_time_min': total_time,
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
        
        return {
            'satellite_stats': sat_stats,
            'target_coverage': target_coverage,
            'total_observations': len(selected_observations),
            'total_data_volume': sum(obs.data_volume for obs in selected_observations)
        }
    
    def visualize_schedule(self, figsize=(15, 10)):
        """Create a Gantt chart visualization of the schedule"""
        if not self.solution or self.solution['status'] != 'Optimal':
            print("No optimal solution to visualize")
            return
        
        selected_obs_idx = self.solution['selected_observations']
        selected_observations = [self.observations[i] for i in selected_obs_idx]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Gantt chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.satellites)))
        satellite_colors = {sat.id: colors[i] for i, sat in enumerate(self.satellites)}
        
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        for satellite in self.satellites:
            sat_obs = [obs for obs in selected_observations if obs.satellite_id == satellite.id]
            sat_obs.sort(key=lambda x: x.start_time)
            
            for obs in sat_obs:
                start_hours = (obs.start_time - selected_observations[0].start_time).total_seconds() / 3600
                duration_hours = obs.duration / 60
                
                target = next(t for t in self.targets if t.id == obs.target_id)
                ax1.barh(y_pos, duration_hours, left=start_hours, 
                        color=satellite_colors[satellite.id], alpha=0.7,
                        label=f"{satellite.name}" if obs == sat_obs[0] else "")
                
                # Add target name annotation
                ax1.text(start_hours + duration_hours/2, y_pos, target.name, 
                        ha='center', va='center', fontsize=8, rotation=45)
            
            y_labels.append(satellite.name)
            y_ticks.append(y_pos)
            y_pos += 1
        
        ax1.set_ylabel('Satellites')
        ax1.set_xlabel('Time (hours)')
        ax1.set_title('Satellite Observation Schedule')
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_labels)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Resource utilization chart
        satellites_names = [sat.name for sat in self.satellites]
        memory_util = []
        
        for satellite in self.satellites:
            sat_obs = [obs for obs in selected_observations if obs.satellite_id == satellite.id]
            total_data = sum(obs.data_volume for obs in sat_obs)
            util = (total_data / satellite.memory_capacity) * 100
            memory_util.append(util)
        
        bars = ax2.bar(satellites_names, memory_util, color=[satellite_colors[sat.id] for sat in self.satellites])
        ax2.set_ylabel('Memory Utilization (%)')
        ax2.set_title('Satellite Memory Utilization')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, util in zip(bars, memory_util):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{util:.1f}%', ha='center', va='bottom')
        
        #plt.tight_layout()
        #plt.show()
        # Save the visualization to a file
        # get script directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        fig.tight_layout()  

        # Save the figure
        plot_path = os.path.join(plot_dir, 'satellite_schedule.png')    
        
        fig.savefig(plot_path, dpi=300) # where? 
        print("Schedule visualization saved as 'satellite_schedule.png'")
        # #get current working directory
        # import os
        # current_dir = os.getcwd()
        # print(f"Current working directory: {current_dir}")

def create_example_scenario():
    """Create an example scenario for testing"""
    
    # Create satellites
    satellites = [
        Satellite(1, "Sentinel-1A", memory_capacity=10.0, power_capacity=200, data_rate=100),
        Satellite(2, "Landsat-9", memory_capacity=15.0, power_capacity=180, data_rate=150),
        Satellite(3, "WorldView-3", memory_capacity=8.0, power_capacity=160, data_rate=120)
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

def main():
    """Main execution function"""
    print("=== Satellite Observation Scheduling using MILP ===\n")
    
    # Create example scenario
    satellites, targets = create_example_scenario()
    
    print(f"Created scenario with {len(satellites)} satellites and {len(targets)} targets")
    
    # Initialize scheduler
    scheduler = SatelliteScheduler(satellites, targets, time_horizon=24)
    
    # Generate observation opportunities
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    observations = scheduler.generate_observation_opportunities(start_time)
    print(f"Generated {len(observations)} observation opportunities")
    
    # Build and solve MILP model
    scheduler.build_milp_model()
    solution = scheduler.solve()
    
    if solution and solution['status'] == 'Optimal':
        # Analyze solution
        analysis = scheduler.analyze_solution()
        
        print("\n=== Solution Analysis ===")
        print(f"Total observations scheduled: {analysis['total_observations']}")
        print(f"Total data volume: {analysis['total_data_volume']:.2f} GB")
        
        print("\nSatellite Statistics:")
        for sat_name, stats in analysis['satellite_stats'].items():
            print(f"  {sat_name}:")
            print(f"    Observations: {stats['observations']}")
            print(f"    Data volume: {stats['total_data_gb']:.2f} GB")
            print(f"    Total time: {stats['total_time_min']:.1f} minutes")
            print(f"    Memory utilization: {stats['memory_utilization']:.1f}%")
        
        print("\nTarget Coverage:")
        for target_name, coverage in analysis['target_coverage'].items():
            status = "✓" if coverage['covered'] else "✗"
            print(f"  {status} {target_name} (Priority: {coverage['priority']:.1f})")
        
        # Visualize results
        scheduler.visualize_schedule()
    
    return scheduler

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    scheduler = main()