import pulp

model = pulp.LpProblem("Waiter_Scheduling_Integer", pulp.LpMinimize)

# Δημιουργία μεταβλητών x1 έως x7
x = [pulp.LpVariable(f"x{i+1}", lowBound=0, cat='Continuous') for i in range(7)]

# Συνάρτηση κόστους: ελαχιστοποίηση συνολικού αριθμού σερβιτόρων
model += pulp.lpSum(x), "Total_Waiters"

# Περιορισμοί ανά ημέρα
model += x[0] + x[3] + x[4] + x[5] + x[6] >= 8   # Δευτέρα
model += x[0] + x[1] + x[4] + x[5] + x[6] >= 8   # Τρίτη
model += x[0] + x[1] + x[2] + x[5] + x[6] >= 8   # Τετάρτη
model += x[0] + x[1] + x[2] + x[3] + x[6] >= 8   # Πέμπτη
model += x[1] + x[2] + x[3] + x[4] + x[5] >= 15  # Παρασκευή
model += x[2] + x[3] + x[4] + x[5] + x[6] >= 15  # Σάββατο
model += x[0] + x[3] + x[4] + x[5] + x[6] >= 10  # Κυριακή

#branch A
#model += x[3] <= 7

#branch A1
# κραταμε το model += x[3] <= 7  # από A
#model += x[1] <= 0  # A1 = x2 <= 0

#branch A11
#model+=x[2]>=6 # x3 >= 6 #optimal solution

#branch A12
#model+=x[2]<=5

# #branch A2
# model += x[3] <= 7  # από A
# model += x[1] >= 1   #optimal solution

# #branch B
# model += x[3] >= 8 #optimal solution





# Λύση με solver CBC
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

# Εμφάνιση αποτελεσμάτων
print("Status:", pulp.LpStatus[model.status])
print("Βέλτιστη Τιμή:", pulp.value(model.objective))
for var in x:
    print(f"{var.name} = {var.varValue}")
