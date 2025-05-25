import pulp

# Model
# Δημιουργία του μοντέλου (ελαχιστοποίηση)
dual = pulp.LpProblem("Dual_LP", pulp.LpMinimize)

# Dual μεταβλητές: y1 για τον περιορισμό βάρους, y2-y6 για τα upper bounds
y_vars = []
for i in range(1, 7):
    y_vars.append(pulp.LpVariable(f'y{i}', lowBound=0, cat='Integer'))

# Αντικειμενική συνάρτηση: min 11*y1 + y2 + y3 + y4 + y5 + y6
y1, y2, y3, y4, y5, y6 = y_vars
dual += 11*y1 + y2 + y3 + y4 + y5 + y6, "Objective"

# Dual constraints για κάθε x_i
dual += 2*y1 + y2 >= 10   # για x1
dual += 3*y1 + y3 >= 14   # για x2
dual += 4*y1 + y4 >= 31   # για x3
dual += 6*y1 + y5 >= 48   # για x4
dual += 8*y1 + y6 >= 60   # για x5

# Επίλυση
dual.solve(pulp.PULP_CBC_CMD(msg=False))

# Εμφάνιση αποτελεσμάτων
print("Status:", pulp.LpStatus[dual.status])
print("Objective value (dual):", pulp.value(dual.objective))
for var in [y1, y2, y3, y4, y5, y6]:
    print(f"{var.name} = {var.varValue:.4f}")


