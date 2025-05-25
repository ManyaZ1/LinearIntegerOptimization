import pulp
import numpy as np

def is_integral(solution, tol=1e-5):
    return all(abs(x - round(x)) <= tol for x in solution)

def branch_and_bound_knapsack():
    profits = [10, 14, 31, 48, 60]
    volumes = [2, 3, 4, 6, 8]
    capacity = 11
    n = len(profits)

    best_value = -np.inf
    best_solution = None
    stack = []
    seen = set()

    x_vars = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=1, cat='Continuous') for i in range(n)]

    def create_model(extra_constraints=None):
        model = pulp.LpProblem("Knapsack_BnB", pulp.LpMaximize)
        model += pulp.lpSum([profits[i] * x_vars[i] for i in range(n)])
        model += pulp.lpSum([volumes[i] * x_vars[i] for i in range(n)]) <= capacity
        if extra_constraints:
            for c in extra_constraints:
                model += c
        return model

    stack.append(([], 0))

    while stack:
        constraints, depth = stack.pop()
        indent = "│   " * depth + "├── "

        model = create_model(constraints)
        glpk_path = r"C:\Users\USER\Documents\_LinearIntegerOptimization\SETS\set2\glpk-4.65\w64\glpsol.exe"
        status = model.solve(pulp.GLPK_CMD(path=glpk_path, msg=0))

        if status != pulp.LpStatusOptimal:
            print(f"{indent}[infeasible or undefined]")
            continue

        sol = [x.varValue for x in x_vars]
        val = pulp.value(model.objective)
        key = tuple(round(v, 3) for v in sol)
        if key in seen:
            print(f"{indent} Skipping duplicate")
            continue
        seen.add(key)

        status_note = "✔" if is_integral(sol) else ""
        print(f"{indent}x = {np.round(sol, 3)}, val = {val:.2f} {status_note}")

        if val <= best_value:
            continue

        if is_integral(sol):
            if val > best_value:
                best_value = val
                best_solution = [int(round(v)) for v in sol]
            continue

        for i in range(n):
            if abs(sol[i] - round(sol[i])) > 1e-5:
                floor_val = np.floor(sol[i])
                ceil_val = np.ceil(sol[i])
                stack.append((constraints + [x_vars[i] <= floor_val], depth + 1))
                stack.append((constraints + [x_vars[i] >= ceil_val], depth + 1))
                break

    return best_value, best_solution

# Εκτέλεση
best_value, best_solution = branch_and_bound_knapsack()

if best_solution is None:
    print("Καμία λύση δεν βρέθηκε.")
else:
    print("\n=== Τελικό αποτέλεσμα ===")
    print("Βέλτιστο Ζ:", best_value)
    print("Επιλεγμένα δέματα:", best_solution)
