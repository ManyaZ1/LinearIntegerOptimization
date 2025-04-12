import numpy as np
import pandas as pd
import itertools

# Πλήθος μεταβλητών: x1, x2, x3, s1, s2, s3, s4 = 7
# Πλήθος εξισώσεων: 4

# Πίνακας συντελεστών A για το σύστημα εξισώσεων μετά την προσθήκη χαλαρώσεων
A = np.array([
    [1, 1, 0, -1,  0,  0,  0],   # x1 + x2 - s1 = 10
    [0, 1, 1,  0, -1,  0,  0],   # x2 + x3 - s2 = 15
    [1, 0, 1,  0,  0, -1,  0],   # x1 + x3 - s3 = 12
    [20, 10, 15, 0, 0, 0, 1]     # 20x1 + 10x2 + 15x3 + s4 = 300
])
b = np.array([10, 15, 12, 300])
var_names = ['x1', 'x2', 'x3', 's1', 's2', 's3', 's4']

basic_solutions = []

# Δοκιμάζουμε όλους τους συνδυασμούς 4 βασικών μεταβλητών από 7
for basic_vars in itertools.combinations(range(7), 4):
    A_basic = A[:, basic_vars]
    try:
        # Λύνουμε το σύστημα για τις βασικές μεταβλητές
        x_basic = np.linalg.solve(A_basic, b)
        
        # Φτιάχνουμε το πλήρες διάνυσμα λύσης
        full_x = np.zeros(7)
        for i, var_idx in enumerate(basic_vars):
            full_x[var_idx] = x_basic[i]
        
        # Ελέγχουμε αν είναι εφικτή λύση (όλες οι μεταβλητές ≥ 0)
        feasible = np.all(full_x >= -1e-6)

        # Εκφυλισμένη: αν κάποια βασική μεταβλητή είναι μηδέν (ή πολύ κοντά)
        degenerate = np.any(np.isclose(x_basic, 0.0, atol=1e-6))
        
        basic_solutions.append({
            'Solution': full_x,
            'Basic Vars': [var_names[i] for i in basic_vars],
            'Feasible': feasible,
            'Degenerate': degenerate
        })

    except np.linalg.LinAlgError:
        continue  # το σύστημα δεν έχει λύση (γραμμικά εξαρτημένο)

# Δημιουργία DataFrame για εμφάνιση
df_basic = pd.DataFrame([{
    **{var_names[i]: sol['Solution'][i] for i in range(7)},
    'Basic Vars': sol['Basic Vars'],
    'Feasible': sol['Feasible'],
    'Degenerate': sol['Degenerate']
} for sol in basic_solutions])

if __name__ == "__main__":
    print("Basic Solutions:")
    print(df_basic)
    print("\nTotal Basic Solutions Found:", len(basic_solutions))
    #degenerate solutions
    print("\nDegenerate Solutions:")
    print(df_basic[df_basic['Degenerate']].to_string(index=False))
    print("\nTotal Degenerate Solutions Found:", len(df_basic[df_basic['Degenerate']]))
# if __name__ == "__main__":
#     print("\n--- Αριθμημένες Βασικές Λύσεις ---")
#     for i, sol in enumerate(basic_solutions):
#         x_vals = sol['Solution'][:3]
#         slack_vals = sol['Solution'][3:]
#         feas = "✓" if sol['Feasible'] else "✗"
#         degen = "✓" if sol['Degenerate'] else "✗"
#         print(f"# {i:2d}: x = {x_vals}, slacks = {slack_vals} | Feasible: {feas}, Degenerate: {degen}")
    
#     print("\nTotal Basic Solutions Found:", len(basic_solutions))
#     print("Total Degenerate Solutions Found:", len(df_basic[df_basic['Degenerate']]))

# Υπολογισμός της αντικειμενικής συνάρτησης Z = 8x1 + 5x2 + 4x3 για κάθε λύση
df_basic["Z"] = 8 * df_basic["x1"] + 5 * df_basic["x2"] + 4 * df_basic["x3"]

# Εύρεση της καλύτερης (εφικτής) λύσης με το ελάχιστο Z
feasible_df = df_basic[df_basic["Feasible"]]
best_idx = feasible_df["Z"].idxmin()  # idxmin επιστρέφει το index της ελάχιστης τιμής
best_vertex = feasible_df.loc[best_idx]

# Δημιουργία DataFrame μόνο με τα σημαντικά
bvf = pd.DataFrame([{
    "x1": best_vertex["x1"],
    "x2": best_vertex["x2"],
    "x3": best_vertex["x3"],
    "Z": best_vertex["Z"]
}])

# print("\nBest vertex:\n")
# print(bvf.to_string(index=False))
# print("\n")

