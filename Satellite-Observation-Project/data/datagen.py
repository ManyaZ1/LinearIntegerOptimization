import json
# Define scenarios of different sizes
scenarios_data = {
    "scenarios": [
        {
            "name": "small",
            "time_horizon": 6,
            "use_conflict_degree": True,
            "satellites": [
                { "id": 1, "name": "NanoSat-1", "memory_capacity": 5.0, "power_capacity": 100, "data_rate": 50, "setup_time": 1.0 }
            ],
            "targets": [
                { "id": 1, "name": "Target_A_small", "lat": 40.0, "lon": -75.0, "priority": 0.8 },
                { "id": 2, "name": "Target_B_small", "lat": 42.0, "lon": -73.0, "priority": 0.6 }
            ]
        },
        {
            "name": "medium",
            "time_horizon": 12,
            "use_conflict_degree": True,
            "satellites": [
                { "id": 1, "name": "Sentinel-1A", "memory_capacity": 10.0, "power_capacity": 200, "data_rate": 100, "setup_time": 2.0 },
                { "id": 2, "name": "Landsat-9", "memory_capacity": 15.0, "power_capacity": 180, "data_rate": 150, "setup_time": 3.0 }
            ],
            "targets": [
                { "id": 1, "name": "Forest_Fire_California", "lat": 37.7749, "lon": -122.4194, "priority": 0.9 },
                { "id": 2, "name": "Crop_Monitoring_Iowa", "lat": 41.5868, "lon": -93.6250, "priority": 0.6 },
                { "id": 3, "name": "Urban_Growth_Tokyo", "lat": 35.6762, "lon": 139.6503, "priority": 0.7 },
                { "id": 4, "name": "Flood_Assessment_Bangladesh", "lat": 23.6850, "lon": 90.3563, "priority": 0.95 }
            ]
        },
        {
            "name": "large",
            "time_horizon": 24,
            "use_conflict_degree": True,
            "satellites": [
                { "id": 1, "name": "Sentinel-1A", "memory_capacity": 10.0, "power_capacity": 200, "data_rate": 100, "setup_time": 2.0 },
                { "id": 2, "name": "Landsat-9", "memory_capacity": 15.0, "power_capacity": 180, "data_rate": 150, "setup_time": 3.0 },
                { "id": 3, "name": "WorldView-3", "memory_capacity": 8.0, "power_capacity": 160, "data_rate": 120, "setup_time": 1.5 }
            ],
            "targets": [
                { "id": 1, "name": "Forest_Fire_California", "lat": 37.7749, "lon": -122.4194, "priority": 0.9 },
                { "id": 2, "name": "Crop_Monitoring_Iowa", "lat": 41.5868, "lon": -93.6250, "priority": 0.6 },
                { "id": 3, "name": "Urban_Growth_Tokyo", "lat": 35.6762, "lon": 139.6503, "priority": 0.7 },
                { "id": 4, "name": "Flood_Assessment_Bangladesh", "lat": 23.6850, "lon": 90.3563, "priority": 0.95 },
                { "id": 5, "name": "Ice_Monitoring_Arctic", "lat": 71.0, "lon": -8.0, "priority": 0.8 },
                { "id": 6, "name": "Desert_Expansion_Sahara", "lat": 23.0, "lon": 8.0, "priority": 0.5 },
                { "id": 7, "name": "Volcano_Activity_Italy", "lat": 40.8518, "lon": 14.2681, "priority": 0.85 }
            ]
        }
    ]
}

# Write the scenarios to a file

import os
cur_dir = os.path.dirname(__file__)
file_path = os.path.join(cur_dir,"scenarios.json")
with open(file_path, "w") as f:
    json.dump(scenarios_data, f, indent=2)

