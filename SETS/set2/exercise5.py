import pulp

# Ορισμός προβλήματος μεγιστοποίησης
model = pulp.LpProblem("Integer_LP_BranchBound", pulp.LpMaximize)

# Μεταβλητές (χαλάρωση: συνεχείς, όχι ακέραιες)
x1 = pulp.LpVariable("x1", lowBound=0, cat='Continuous')
x2 = pulp.LpVariable("x2", lowBound=0, cat='Continuous')
x3 = pulp.LpVariable("x3", lowBound=0, cat='Continuous')
# x1 = pulp.LpVariable("x1", lowBound=0, cat='Integer')
# x2 = pulp.LpVariable("x2", lowBound=0, cat='Integer')
# x3 = pulp.LpVariable("x3", lowBound=0, cat='Integer')

# Αντικειμενική συνάρτηση
model += 34 * x1 + 29 * x2 + 2 * x3

# Περιορισμοί
model += 7 * x1 + 5 * x2 - x3 <= 16
model += -x1 + 3 * x2 + x3 <= 10
model += -x2 + 2 * x3 <= 3

# # branch A
model += x2 <= 2
# print("x2 <= 2")

# # branch A1
model += x3 <= 2
# print("x3 <= 2")

# # branch A11
model += x1 <= 1 ## optimal solution
# print("x1 <= 1") ## optimal solution

# # branch A12
# model += x1>=2 ## optimal solution
# print("x1 >=2") ## optimal solution

# # branch A2
#model += x3 >=3
#print("x3 >= 3")

# branch A21
#model += x1 <= 1
#print("x1 <= 1") 

# branch A22
# model += x1 >= 2 
# print("x1 >= 2 ") 



# branch B
#model += x2 >= 3
# print("x2 >= 3")

# # branch B1
#model += x3 <= 1
#print("x3 <= 1")

# # branch B11
#model += x2 <= 3
#print("x2 <= 3")

# # branch B111
# model += x1 <= 0
# print("x1 <= 0")

# # branch B112
# model += x1 >= 1
# print("x1 >= 1")



# # branch B12
# model += x2 >=4
# print("x2 >=4")

# # branch B2
# model += x3 >= 2
# print("x3 >= 2")

# Επίλυση
solver = pulp.PULP_CBC_CMD(msg=False)
model.solve(solver)

# Εμφάνιση αποτελεσμάτων

print("(Z):", round(pulp.value(model.objective),2))
print(f"x1 = {x1.varValue}")
print(f"x2 = {x2.varValue}")
print(f"x3 = {x3.varValue}")
print("Status:", pulp.LpStatus[model.status])