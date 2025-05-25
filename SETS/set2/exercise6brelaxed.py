import pulp
import numpy as np

profits = [10, 14, 31, 48, 60]
volumes = [2, 3, 4, 6, 8]
capacity = 11
n = len(profits)
x_vars = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=1, cat='Continuous') for i in range(n)]

model = pulp.LpProblem("Knapsack_BnB", pulp.LpMaximize)
model += pulp.lpSum([profits[i] * x_vars[i] for i in range(n)])
model += pulp.lpSum([volumes[i] * x_vars[i] for i in range(n)]) <= capacity

model.solve(pulp.PULP_CBC_CMD(msg=0))
print("✅ Βέλτιστη ακέραια λύση του προβλήματος:")
for i in range(n):
    print(f"x{i+1} =", x_vars[i].varValue, end=", ") 
print("Objective =", pulp.value(model.objective))
print("capacity filled =", sum(volumes[i] * x_vars[i].varValue for i in range(n)))
print("status =", pulp.LpStatus[model.status])
