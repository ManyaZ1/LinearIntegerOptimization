import pulp
import numpy as np


def exercise1_model():
    # Δημιουργία μοντέλου
    model = pulp.LpProblem("LP1", pulp.LpMaximize)

    # Δημιουργία μεταβλητών
    x1 = pulp.LpVariable('x1', cat='Continuous')  # x1 ∈ R
    x2 = pulp.LpVariable('x2', lowBound=0)
    x3 = pulp.LpVariable('x3', lowBound=0)
    x4 = pulp.LpVariable('x4', lowBound=0)
    x5 = pulp.LpVariable('x5', lowBound=0)

    # Αντικειμενική συνάρτηση
    model += 3*x1 + 11*x2 + 9*x3 - x4 - 29*x5, "Objective"

    # Περιορισμοί
    model += x2 + x3 + x4 - 2*x5 <= 4, "Constraint_1"
    model += x1 - x2 + x3 + 2*x4 + x5 >= 0, "Constraint_2"
    model += x1 + x2 + x3 - 3*x5 <= 1, "Constraint_3"

    # Επίλυση
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model

def extract_basic_matrix(A, b, variable_values, var_to_index):
    basic_vars = [v for v, val in variable_values.items() if abs(val) > 1e-8]
    nonbasic_vars = [v for v in variable_values if v not in basic_vars]

    B = A[:, [var_to_index[v] for v in basic_vars]]
    N = A[:, [var_to_index[v] for v in nonbasic_vars]]

    xB = np.linalg.solve(B, b)

    print("Βασικές μεταβλητές:", basic_vars)
    print("Μη βασικές μεταβλητές:", nonbasic_vars)
    print("Πίνακας B:")
    print(B)
    print("B_inv")
    B_inv = np.linalg.inv(B)
    print(B_inv)
    
    print("b=", b)

    print("xB =", xB)

    #e=np.array([1.,0.,0.])
    #print("DF",B_inv @ e)
    return B, xB, N, basic_vars, nonbasic_vars    




def answer_a(model):
    print("Status:", pulp.LpStatus[model.status])
    print("Objective value:", pulp.value(model.objective))
    for var in model.variables():
        print(f"{var.name} = {var.varValue}")

    variable_values = {var.name: var.varValue for var in model.variables()}

    print("Μεταβλητή χαλάρωσης (slack) του περιορισμού 1 (<=):", model.constraints["Constraint_1"].slack)
    print("Μεταβλητή πλεονάσματος (surplus) του περιορισμού 2 (>=):", model.constraints["Constraint_2"].slack)
    print("Μεταβλητή χαλάρωσης (slack) του περιορισμού 3 (<=):", model.constraints["Constraint_3"].slack)

    A = np.array([
        [0, 1, 1, 1, -2],  # Constraint 1
        [1, -1, 1, 2, 1],  # Constraint 2
        [1, 1, 1, 0, -3],  # Constraint 3
    ])
    b = np.array([4, 0, 1])

    var_to_index = {var: i for i, var in enumerate(variable_values.keys())}
    c = {
        'x1': 3,
        'x2': 11,
        'x3': 9,
        'x4': -1,
        'x5': -29
    }

    B, xB, N, basic_vars, nonbasic_vars = extract_basic_matrix(A, b, variable_values, var_to_index)

def answer_d():
    # Create LP
    dual = pulp.LpProblem("Dual", pulp.LpMinimize)

    # Variables
    y1 = pulp.LpVariable('y1', lowBound=0)
    y2 = pulp.LpVariable('y2', upBound=0)
    y3 = pulp.LpVariable('y3', lowBound=0)

    # Objective
    dual += 4*y1 + 0*y2 + 1*y3

    # Constraints
    dual += y2 + y3 == 3
    dual += y1 - y2 + y3 >= 11
    dual += y1 + y2 + y3 >= 9
    dual += y1 + 2*y2 >= -1
    dual += -2*y1 + y2 - 3*y3 >= -29

    # Solve
    dual.solve(pulp.PULP_CBC_CMD(msg=False))
    print("Status:", pulp.LpStatus[dual.status])
    for var in dual.variables():
        print(f"{var.name} = {var.varValue}")
    print("Objective =", pulp.value(dual.objective))    

if __name__ == "__main__":
    model = exercise1_model()
    answer_a(model)
    # Μετά τη λύση του μοντέλου:
    variable_values = {var.name: var.varValue for var in model.variables()}
    var_to_index = {var: i for i, var in enumerate(variable_values.keys())}
    basic_vars = [v for v, val in variable_values.items() if abs(val) > 1e-8]
    nonbasic_vars = [v for v in variable_values if v not in basic_vars]

    # Πίνακας A και b
    A = np.array([
        [0, 1, 1, 1, -2],
        [1, -1, 1, 2, 1],
        [1, 1, 1, 0, -3]
    ])
    b = np.array([4, 0, 1])

    # Δημιουργία B και N
    B = A[:, [var_to_index[v] for v in basic_vars]]
    N = A[:, [var_to_index[v] for v in nonbasic_vars]]
    B_inv = np.linalg.inv(B)
    print("----------------------------------------")
    print("                  DUAL")
    print("----------------------------------------")
    answer_d()





    
    











