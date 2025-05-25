import pulp

# Δημιουργία προβλήματος
model = pulp.LpProblem("Dual_LP", pulp.LpMaximize)

# Μεταβλητές: y1 ≤ 0, y3 ≥ 0, y2 ελεύθερη
y1 = pulp.LpVariable("y1", upBound=0)
y2 = pulp.LpVariable("y2")  # unrestricted
y3 = pulp.LpVariable("y3", lowBound=0)

# Αντικειμενική συνάρτηση: max 6y2 + 3y3
model += 6*y2 + 3*y3, "Objective"

# Περιορισμοί
model += 2*y1 - y2 + 3*y3 >= 1    # (x1 ≤ 0)
model += 3*y1 + y2 + y3 <= 1      # (x2 ≥ 0)
model += y1 + 2*y2 + 4*y3 == 0    # (x3 ∈ ℝ)
model += y1 + y2 + 2*y3 <= 0      # (x4 ≥ 0)

# Επίλυση
model.solve(pulp.PULP_CBC_CMD(msg=False))

# Αποτελέσματα
print("Status:", pulp.LpStatus[model.status])
print("Objective value:", pulp.value(model.objective))
for var in model.variables():
    print(f"{var.name} = {var.varValue}")
