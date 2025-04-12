import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# Επανακαθορισμός όλων των περιορισμών ως γραμμές
x = np.linspace(-10, 10, 400)

# Περιορισμοί
def line1(x): return 4 - 2 * x       # 6x1 + 3x2 = 12
def line2(x): return (4 - x) / 2     # 4x1 + 8x2 = 16
def line3(x): return (30 - 6*x) / 5  # 6x1 + 5x2 = 30
def line4(x): return (36 - 6*x) / 7  # 6x1 + 7x2 = 36
def line5(x): return 0 * x            # Άξονας x1


def intersection(A1, b1, A2, b2): # εξισώσεις με μορφή Ax = b ευρεση σημείων τομής
    A = np.array([A1, A2])
    b = np.array([b1, b2])
    try:
        sol = np.linalg.solve(A, b)
        return tuple(sol)
    except np.linalg.LinAlgError:
        return None
    
# Συντελεστές περιορισμών A και b 
#x1: x2=0 άρα Α=[0,1] και b=0 για αξονα x1
#x2: x1=0 άρα Α=[1,0] και b=0 για αξονα x2
coefsA=[[6,3],[4,8],[6,5],[6,7],[0,1],[1,0]]
coefsB=[12,16,30,36,0,0]
# Υπολογισμός τομών μεταξύ περιορισμών και σχεδιασμός σημείων τομής
intersections = []
for i in range(len(coefsA)):
    for j in range(i+1, len(coefsA)):
        x_val, y_val = intersection(coefsA[i], coefsB[i], coefsA[j], coefsB[j])
        if x_val is not None and y_val is not None:
            
            
            # Έλεγχος αν οι τομές είναι στον 1ο τεταρτημόριο    
            if x_val >= 0 and y_val >= 0:
                # Έλεγχος αν οι τομές ικανοποιούν όλους τους περιορισμούς
                if (
                    2*x_val + y_val >= 4 and
                    x_val + 2*y_val >= 4 and
                    6*x_val + 5*y_val <= 30 and
                    6*x_val + 7*y_val <= 36
                ):
                    intersections.append([x_val, y_val])
                    #plt.plot(x_val, y_val, 'ro')
                    #plt.text(x_val + 0.1, y_val, f'P{i+1}{j+1}')

intersections = np.array(intersections) #valid intersection
########### - χρωματισμός εφικτής περιοχής  #######################
######################################################### εύρεση κυρτού περιβλήματος ##############################################################
# Υπολογισμός κέντρου
center = np.mean(intersections, axis=0)
# Υπολογισμός γωνίας κάθε σημείου σε σχέση με το κέντρο
angles = np.arctan2(intersections[:,1] - center[1], intersections[:,0] - center[0])
# Ταξινόμηση των σημείων με βάση τις γωνίες
sorted_indices = np.argsort(angles)
sorted_vertices = intersections[sorted_indices]

# Με γραφικό τρόπο βρείτε  τη βέλτιστη κορυφή του προβλήματος, εάν υπάρχει. # Υπολογισμός Ζ
#slider
# Ζ = 3x1 + x2
# Υπολογισμός Ζ για κάθε σημείο
def Z(x1, x2): return 3 * x1 + x2
z_values = [Z(x, y) for x, y in intersections]

# Δημιουργία γραφήματος
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

# Σχεδίαση περιορισμών
ax.plot(x, line1(x), label='6x1 + 3x2 = 12')
ax.plot(x, line2(x), label='4x1 + 8x2 = 16')
ax.plot(x, line3(x), label='6x1 + 5x2 = 30')
ax.plot(x, line4(x), label='6x1 + 7x2 = 36')
ax.plot(x, line5(x), label='x1 (x2 = 0)')
ax.plot(0 * x, x, label='x2 (x1 = 0)')
ax.axhline(0, color='black', lw=0.5)

# Σχεδίαση εφικτής περιοχής
ax.fill(sorted_vertices[:, 0], sorted_vertices[:, 1], color='lightgreen', alpha=0.4, label='Εφικτή περιοχή')

# Σχεδίαση κορυφών
for i, (xv, yv) in enumerate(intersections):
    ax.plot(xv, yv, 'ro')
    ax.text(xv + 0.1, yv, f'P{i+1}', fontsize=9)
    print(f'Κορυφή P{i+1}: {xv}, {yv}')

# Προσθήκη ευθείας Z
initial_z = 8
z_line, = ax.plot(x, initial_z - 3 * x, 'r--', label='Z = 3x₁ + x₂')  #x2=z-3x1

def currentvertex(z_val):
    z_val = z_slider.val
    tol=0.1
    for x, y in intersections:
        if abs(Z(x, y) - z_val) <= tol:
            #print(f'Κορυφή: {x}, {y}')
            return np.array([x, y])
    return None
highlight, = ax.plot([], [], 'mo', markersize=10, label='Κοντινή κορυφή')

# Slider
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
z_slider = Slider(ax_slider, 'Z', 0, 20, valinit=initial_z, valstep=0.1)

# Slider update function
def update(val):
    z_val = z_slider.val
    z_line.set_ydata(z_val - 3 * x)
    bv = currentvertex(z_val)
    if bv is not None:
        highlight.set_data([bv[0]], [bv[1]])
    else:
        highlight.set_data([], [])
    fig.canvas.draw_idle()


z_slider.on_changed(update)
update(initial_z)

ax.set_xlim(-2, 7)
ax.set_ylim(-2, 7)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Γραφική εύρεση βέλτιστης κορυφής με slider αντικειμενικής συνάρτησης Z')
ax.legend()
plt.grid(True)
plt.show()

