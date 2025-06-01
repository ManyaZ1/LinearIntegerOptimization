from pulp import *

# --- Sets ---
satellites = ['S1', 'S2']
targets = ['T1', 'T2', 'T3', 'T4', 'T5']
times = list(range(6))  # 0 to 5

# --- Parameters: visibility[s][t][k] = 1 if satellite s can observe target t at time k ---
visibility = {
    'S1': {
        'T1': [1, 1, 0, 0, 0, 0],
        'T2': [0, 1, 1, 0, 0, 0],
        'T3': [0, 0, 1, 1, 0, 0],
        'T4': [0, 0, 0, 1, 1, 0],
        'T5': [0, 0, 0, 0, 1, 1],
    },
    'S2': {
        'T1': [0, 0, 1, 1, 0, 0],
        'T2': [0, 0, 0, 1, 1, 0],
        'T3': [0, 0, 0, 0, 1, 1],
        'T4': [1, 1, 0, 0, 0, 0],
        'T5': [0, 1, 1, 0, 0, 0],
    }
}

# --- Define MILP Model ---
model = LpProblem("Satellite_Observation_Scheduling", LpMaximize)

# --- Decision variables: x[s,t,k] ∈ {0,1} if satellite s observes target t at time k ---
x = LpVariable.dicts("x", 
    ((s, t, k) for s in satellites for t in targets for k in times),
    cat=LpBinary
)

# --- Objective: maximize number of observations ---
model += lpSum(x[s, t, k] 
               for s in satellites 
               for t in targets 
               for k in times 
               if visibility[s][t][k] == 1)

# --- Constraints ---

# (1) One observation per target at most (regardless of who does it and when)
for t in targets:
    model += lpSum(x[s, t, k] 
                   for s in satellites 
                   for k in times 
                   if visibility[s][t][k] == 1) <= 1

# (2) One observation per satellite per time
for s in satellites:
    for k in times:
        model += lpSum(x[s, t, k] 
                       for t in targets 
                       if visibility[s][t][k] == 1) <= 1

# (3) Enforce visibility
for s in satellites:
    for t in targets:
        for k in times:
            if visibility[s][t][k] == 0:
                model += x[s, t, k] == 0

# --- Solve with GLPK ---
model.solve(GLPK(msg=True,path=r'C:\Users\USER\Documents\GitHub\LinearIntegerOptimization\Satellite-Observation-Project\glpk-4.65\w64\glpsol.exe'))

# --- Output solution ---
print(f"Status: {LpStatus[model.status]}")
print("Selected observations:")
for s in satellites:
    for t in targets:
        for k in times:
            if value(x[s, t, k]) == 1:
                print(f"  Satellite {s} observes {t} at time {k}")
