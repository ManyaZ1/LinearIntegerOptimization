# Άσκηση 6 (β)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Ορισμός προβλήματος (ίδιο με 6α)
A = np.array([
    [1, 2, 4, -1, 1, 0, 0],
    [2, 3, -1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
], dtype=float)

b = np.array([6, 12, 2], dtype=float)
c = np.array([2, 1, 6, -4, 0, 0, 0], dtype=float)

var_names = ["x1", "x2", "x3", "x4", "s1", "s2", "s3"]

# Αρχικοποίηση
initial_tableau = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
initial_tableau[:-1, :-1] = A
initial_tableau[:-1, -1] = b
initial_tableau[-1, :-1] = c
initial_tableau[-1, -1] = 0
initial_basis = [4, 5, 6]

# Κόμβοι: καταστάσεις με (basis, values, Z)
G = nx.DiGraph()
visited = set()


def tableau_key(basis, tableau):
    return tuple(basis), tuple(np.round(tableau[:, -1], 4)), round(tableau[-1, -1], 4)


def explore(tableau, basis):
    # Δημιουργούμε ένα μοναδικό κλειδί για την τρέχουσα κατάσταση (βάση και τιμές b + Z)
    key = tableau_key(basis, tableau)
    # Αν το έχουμε ήδη επισκεφθεί, δεν το ξαναεπεξεργαζόμαστε
    if key in visited:
        return
    visited.add(key)
    # ετικέτα για τον κόμβο
    label = f"B={[var_names[i] for i in basis]}\n"
    label += f"x={tuple(round(tableau[i, -1], 2) for i in range(len(basis)))}\n"
    label += f"Z={round(tableau[-1, -1], 2)}" 
    # Προσθέτουμε τον κόμβο στον γράφο G
    # Αν η γραμμή Z έχει όλα τα στοιχεία ≤ 0 → είναι βέλτιστη λύση (χρωματίζεται πορτοκαλί)
    G.add_node(key, label=label, color="#ffa724" if np.all(tableau[-1, :-1] <= 0) else "#42f5b6")
    # υποψήφιες εισερχόμενες μεταβλητές = αυτές με θετικό συντελεστή στο Z
    last_row = tableau[-1, :-1]
    entering_vars = [j for j in range(len(last_row)) if last_row[j] > 0] # Εξετάζουμε τις υποψήφιες εισερχόμενες μεταβλητές
    for entering in entering_vars:
        # κάνουμε το κριτήριο ελαχίστου λόγου
        ratios = []
        for i in range(len(tableau) - 1):
            aij = tableau[i, entering]
            if aij > 0:
                ratios.append((tableau[i, -1] / aij, i))
        if not ratios:# Αν δεν υπάρχει έγκυρη επιλογή, πάμε στο επόμενο entering
            continue

        min_ratio = min(ratios)[0]
         # Επιλέγουμε όλες τις γραμμές που έχουν min_ratio (ισοπαλία => πολλαπλές πορείες)
        for ratio, leaving_row in ratios:
            if np.isclose(ratio, min_ratio):
                new_tableau = tableau.copy()
                pivot = new_tableau[leaving_row, entering]# κάνουμε κανονικοποίηση της γραμμής εξερχόμενης μεταβλητής
                new_tableau[leaving_row] /= pivot
                for i in range(len(new_tableau)):
                    if i != leaving_row:
                        new_tableau[i] -= new_tableau[i, entering] * new_tableau[leaving_row]

                new_basis = basis.copy()
                # Ενημερώνουμε τη βάση: η εισερχόμενη μεταβλητή μπαίνει στη θέση της εξερχόμενης
                new_basis[leaving_row] = entering
                new_key = tableau_key(new_basis, new_tableau) ## Δημιουργούμε νέο μοναδικό κλειδί για τον επόμενο κόμβο
                # Προσθέτουμε ακμή στον γράφο από την προηγούμενη κατάσταση στη νέα
                G.add_edge(key, new_key, label=f"{var_names[entering]} / {var_names[basis[leaving_row]]}")
                explore(new_tableau, new_basis)  # Επαναληπτικά εξερευνούμε τον νέο κόμβο


explore(initial_tableau.copy(), initial_basis.copy())

# Σχεδίαση γράφου
pos = nx.spring_layout(G, seed=42)
node_colors = [G.nodes[n]["color"] for n in G.nodes]
labels = nx.get_node_attributes(G, "label")
edge_labels = nx.get_edge_attributes(G, "label")

plt.figure(figsize=(14, 8))
nx.draw_networkx_nodes(G, pos, node_color = node_colors, edgecolors = "black", node_size = 7000) 
nx.draw_networkx_edges(G, pos) 
nx.draw_networkx_labels(G, pos, labels=labels, font_size = 10) 
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size = 9) 
plt.title("Simplex Adjacency Graph") 
plt.axis("off") 
plt.tight_layout() 
plt.show()
