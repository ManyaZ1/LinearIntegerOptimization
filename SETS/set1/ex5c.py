from ex5 import all_vertices
from ex5b import basic_solutions
import pandas as pd
import numpy as np


print("\n--- Matching All Basic Feasible Solutions to Vertices ---")
#print(basic_solutions)
#print(all_vertices)
#remone Nan values from all_vertices
all_vertices = np.array(all_vertices)
all_vertices = all_vertices[~np.isnan(all_vertices).any(axis=1)]
#remone Nan values from basic_solutions
basic_solutions = np.array([sol['Solution'] for sol in basic_solutions])
basic_solutions = basic_solutions[~np.isnan(basic_solutions).any(axis=1)]
for i, sol in enumerate(basic_solutions):
    for j, vertex in enumerate(all_vertices):
        if np.allclose(sol[:3], vertex, atol=1e-6):

            print(f"Basic feasible solution #{i} matches vertex #{j}: {vertex}")


######### ΒΕΛΤΙΣΤΗ ΛΥΣΗ #########
from ex5b import df_basic
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
print("\nBest vertex:\n")
print(bvf.to_string(index=False))
print("\n")