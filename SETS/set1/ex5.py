'''min Z = 8x1+5x2+4x3
 x1 +x2 ≥ 10
 x2 +x3 ≥ 15
 x1 +x3 ≥ 12
 20x1 +10x2 +15x3 ≤ 300
 x1, x2,x3 ≥ 0
'''
import numpy as np
import itertools
import pandas as pd

# Ορίζουμε τους περιορισμούς ως εξισώσεις της μορφής Ax = b
# έχουμε 7 περιορισμούς: της μορφής Αχ<=β
# (1) x1 + x2 >= 10     → -x1 - x2 <= -10
# (2) x2 + x3 >= 15     → -x2 - x3 <= -15
# (3) x1 + x3 >= 12     → -x1 - x3 <= -12
# (4) 20x1 + 10x2 + 15x3 <= 300
# (5) x1 >= 0           → -x1 <= 0
# (6) x2 >= 0           → -x2 <= 0
# (7) x3 >= 0           → -x3 <= 0

A = np.array([
    [-1, -1,  0],  # (1)
    [ 0, -1, -1],  # (2)
    [-1,  0, -1],  # (3)
    [20, 10, 15],  # (4)
    [-1,  0,  0],  # (5)
    [ 0, -1,  0],  # (6)
    [ 0,  0, -1]   # (7)
])
b = np.array([-10, -15, -12, 300, 0, 0, 0])

# Δημιουργούμε όλα τα δυνατά συστήματα 3 περιορισμών (συνδυασμοί 3 από 7)
# και αποθηκεύουμε ΚΑΙ τις μη εφικτές ή μη ορισμένες λύσεις 

all_combinations = list(itertools.combinations(range(7), 3)) # επιλέγουμε 3 από 7 περιορισμούς (παίρνουμε όλους τους συνδυασμούς τριάδων αριθμών 0 εως 6)

# Λίστες για πλήρη καταγραφή όλων των συνδυασμών
all_vertices = []
all_statuses = []  # 'feasible', 'infeasible', 'singular'
all_Z = []
fes_vertices=[] # Λίστα για εφικτές κορυφές
fes_Z=[] # Λίστα για εφικτές τιμές Z
for indices in all_combinations:
    A_eq = A[list(indices)]
    b_eq = b[list(indices)]
    try:
        sol = np.linalg.solve(A_eq, b_eq) # Λύνουμε το σύστημα των 3 εξισώσεων
        # Ελέγχουμε αν ικανοποιεί όλους τους περιορισμούς (εφικτή κορυφή)
        if np.all(A @ sol <= b + 1e-6):
            status = "feasible"
            fes_vertices.append(sol) # Αποθηκεύουμε την εφικτή κορυφή
            #κάνουμε έλεγχο για το αν είναι εφικτή η λύση Α * sol <=b ελέγχει όλους τους περιορισμούς περασμένους στον πίνακα Α	
        else:
            status = "infeasible"
        Z_val = 8 * sol[0] + 5 * sol[1] + 4 * sol[2] # Υπολογίζουμε την τιμή του Z
        all_vertices.append(sol)
        all_statuses.append(status)
        all_Z.append(Z_val)
        if status == "feasible":
             fes_Z.append(Z_val) # Αποθηκεύουμε την εφικτή τιμή Z
    except np.linalg.LinAlgError:
        # Σύστημα μη αντιστρέψιμο (γραμμικά εξαρτημένο)
        all_vertices.append([np.nan, np.nan, np.nan])
        all_statuses.append("singular")
        all_Z.append(np.nan)

# Τώρα βρίσκουμε για κάθε κορυφή πόσοι περιορισμοί είναι ενεργοί
degenerate_flags = []
Z_values = []

for v in all_vertices :
    # Ένας περιορισμός είναι ενεργός αν A[i] @ x == b[i]
    active = np.isclose(A @ v, b, atol=1e-6) # ελέγχουμε αν είναι κοντά στην τιμή b
    # αντίστοιχο του active = np.array([np.isclose(A[i] @ v, b[i], atol=1e-6) for i in range(len(A))]) η numpy είναι πιο γρήγορη 
    num_active = np.sum(active) #   active = διάνυσμα boolean τιμών, άρα num_active = πλήθος True τιμών 
    degenerate_flags.append(num_active > 3) # αν είναι περισσότεροι από 3 τότε είναι εκφυλισμένο
    # Υπολογισμός Z = 8x1 + 5x2 + 4x3
    Z = 8*v[0] + 5*v[1] + 4*v[2]
    Z_values.append(Z)
# Δημιουργία DataFrame με όλα τα αποτελέσματα
df_all = pd.DataFrame(all_vertices, columns=["x1", "x2", "x3"])
df_all["Constraints"] = all_combinations
df_all["Status"] = all_statuses
df_all["Degenerate"] = degenerate_flags
df_all["Z"] = all_Z


# Δημιουργούμε πίνακα αποτελεσμάτων
df = pd.DataFrame(fes_vertices, columns=["x1", "x2", "x3"])
df["Z"] = fes_Z
if __name__=="__main__":
    print("\n Κορυφές:")
    print(df_all)
    print("\n\n")
    print("Vertices of the feasible region:")
    print(df)