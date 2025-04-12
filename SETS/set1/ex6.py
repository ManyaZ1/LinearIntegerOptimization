import numpy as np

# Problem definition
# max Z = 2x1 + x2 + 6x3 - 4x4
# Subject to:
# x1 + 2x2 + 4x3 - x4       ≤ 6
# 2x1 + 3x2 -  x3 + x4      ≤ 12
# x1        +  x3 + x4      ≤ 2

# Add slack variables: s1, s2, s3
A = np.array([
    [1, 2, 4, -1, 1, 0, 0],
    [2, 3, -1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
], dtype=float)

b = np.array([6, 12, 2], dtype=float)
c = np.array([2, 1, 6, -4, 0, 0, 0], dtype=float)  # Objective function

'''Εισάγονται μεταβλητές χαλάρωσης και δημιουργείται μία πρώτη βασική 
εφικτή λύση: (x = 0,xS = b)
'''
# Create initial tableau
tableau = np.zeros((A.shape[0] + 1, A.shape[1] + 1)) #rows + 1, columns + 1
tableau[:-1, :-1] = A # Coefficients of constraints
tableau[:-1, -1] = b # Right-hand side 
tableau[-1, :-1] = c # Coefficients of objective function
tableau[-1, -1] = 0 # Objective function value

basic_vars = [4, 5, 6]  # Slack variables initially in basis

def print_tableau(tab, basic_vars, step_desc):
    print("\n" + "-"*80)
    print(step_desc)
    print("-"*80)
    header = ["x1", "x2", "x3", "x4", "s1", "s2", "s3", "b"]
    print("    | " + " | ".join(f"{h:>6}" for h in header))
    print("="*80)
    print("-Z  | " + " | ".join(f"{v:6.2f}" for v in tab[-1]))
    print("-"*80)
    for i in range(len(tab) - 1):
        var_names = ["x1", "x2", "x3", "x4", "s1", "s2", "s3"]
        row_label = var_names[basic_vars[i]]

        #row_label = f"x{basic_vars[i]+1}"
        print(f"{row_label:>3} | " + " | ".join(f"{v:6.2f}" for v in tab[i]))
    print("-"*80)
    # print("-Z  | " + " | ".join(f"{v:6.2f}" for v in tab[-1]))
    # print("-"*80)

def simplex(tableau, basic_vars):
    iteration = 0
    while True:
        print_tableau(tableau, basic_vars, f"Step {iteration}")

        last_row = tableau[-1, :-1]
        if np.all(last_row <= 0):
            print("Βρέθηκε βέλτιστη λύση (όλοι οι συντελεστές Z ≤ 0).")
            break


        # Step 1: Entering variable
        entering_candidates = np.where(last_row > 0)[0]
        entering = entering_candidates[np.argmax(last_row[entering_candidates])] # is index of the entering variable

        print(f"Εισερχόμενη μεταβλητή: x{entering + 1}")

        # Step 2: Minimum ratio test
        ratios = []
        for i in range(len(tableau) - 1):
            col_val = tableau[i, entering]
            if col_val > 0:
                ratios.append(tableau[i, -1] / col_val)
            else:
                ratios.append(np.inf)

        leaving_row = np.argmin(ratios)
        if ratios[leaving_row] == np.inf: # αν ολα αρνητικά => unbounded
            print(" Το πρόβλημα είναι μη φραγμένο.")
            break

        pivot = tableau[leaving_row, entering]
        print(f"Εξερχόμενη μεταβλητή: x{basic_vars[leaving_row] + 1}")
        print(f"Pivot στοιχείο: {pivot:.2f}")

        # Normalize pivot row
        tableau[leaving_row] /= pivot

        # Eliminate pivot column from other rows
        for i in range(len(tableau)):
            if i != leaving_row:
                tableau[i] -= tableau[i, entering] * tableau[leaving_row]

        basic_vars[leaving_row] = entering
        iteration += 1
    return tableau, basic_vars

# Run the algorithm
# Run the algorithm
final_tableau, final_basic_vars = simplex(tableau.copy(), basic_vars.copy())

# Ονόματα μεταβλητών (με βάση τις στήλες)
var_names = ["x1", "x2", "x3", "x4", "s1", "s2", "s3"]

# Αρχικοποίηση λύσης με μηδενικά
solution = {var: 0.0 for var in var_names}

# Για κάθε βασική μεταβλητή, πάρε την τιμή από τη στήλη b (τελευταία)
for i, var_index in enumerate(final_basic_vars):
    var_name = var_names[var_index]
    solution[var_name] = final_tableau[i, -1]

# Εκτύπωση τιμών μεταβλητών
print("Βέλτιστη λύση:")
for var in var_names:
    print(f"{var} = {solution[var]:.2f}")
    
# Τιμή της αντικειμενικής συνάρτησης Z
print(f"Z = {-1*final_tableau[-1, -1]:.2f}")


