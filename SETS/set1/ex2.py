import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# Επανακαθορισμός όλων των περιορισμών ως γραμμές
x = np.linspace(-1, 20, 400)

# Περιορισμοί
# 0.3x1 + 0.1x2 <= 2.7
# 0.5x1 + 0.5x2 = 6
# 0.6x1 + 0.4x2 >=6
def line1(x): return 27. - 3. * x       # 0.3*x1 + 0.1*x2 <= 2.7
def line2(x): return 12.-x     # 0.5*x1 + 0.5*x2 = 6
def line3(x): return 15.-3.*x/2  # 0.6*x1 + 0.4*x2 >=6

def line5(x): return 0 * x            # Άξονας x1

def intersection(A1, b1, A2, b2): # εξισώσεις με μορφή Ax = b
    A = np.array([A1, A2])
    b = np.array([b1, b2])
    try:
        sol = np.linalg.solve(A, b)
        return tuple(sol)
    except np.linalg.LinAlgError:
        return None
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)    
# Συντελεστές περιορισμών A και b 
#x1: x2=0 άρα Α=[0,1] και b=0
#x2: x1=0 άρα Α=[1,0] και b=0   
coefsA=[[0.3,0.1],[0.5,0.5],[0.6,0.4],[0,1],[1,0]]
coefsB=[2.7, 6, 6, 0, 0]
# Υπολογισμός τομών μεταξύ περιορισμών και σχεδιασμός σημείων τομής
intersections = []
for i in range(len(coefsA)):
    for j in range(i+1, len(coefsA)):
        x_val, y_val = intersection(coefsA[i], coefsB[i], coefsA[j], coefsB[j])
        #print(f'Intersection {i+1} and {j+1}: {x_val}, {y_val}')

        # if( i==0 and j==1):
        #     print(f'Intersection {i+1} and {j+1}: {x_val}, {y_val}')
        #     print(round(0.3*x_val + 0.1*y_val, 3) )#<= 2.7
        #     print(isclose(0.5*x_val + 0.5*y_val, 6.0 ))
        #     print(0.6*x_val + 0.4*y_val >=6)

        if x_val is not None and y_val is not None:
            
            
            # Έλεγχος αν οι τομές είναι στον 1ο τεταρτημόριο    
            if x_val >= 0 and y_val >= 0:
                # Έλεγχος αν οι τομές ικανοποιούν όλους τους περιορισμούς
                if (
                    round(0.3*x_val + 0.1*y_val,3) <= 2.7 and
                    isclose(0.5*x_val + 0.5*y_val, 6.0 )and
                    0.6*x_val + 0.4*y_val >=6):

                    intersections.append([x_val, y_val])
                    #print(f'Intersection {i+1} and {j+1}: {x_val}, {y_val}, Z:{}')
                    #plt.plot(x_val, y_val, 'ro')
                    #plt.text(x_val + 0.1, y_val, f'P{i+1}{j+1}')
# Υπολογισμός Ζ για κάθε σημείο
def Z(x1, x2): return ((0.4 * x1 + 0.5*x2)*1.)
# Retrieve the second row #row = arr[1, :]
intersections = np.array(intersections) #valid intersection

######################################################### εύρεση κυρτού περιβλήματος ##############################################################
# Υπολογισμός κέντρου
center = np.mean(intersections, axis=0)
# Υπολογισμός γωνίας κάθε σημείου σε σχέση με το κέντρο
angles = np.arctan2(intersections[:,1] - center[1], intersections[:,0] - center[0])
# Ταξινόμηση των σημείων με βάση τις γωνίες
sorted_indices = np.argsort(angles)
sorted_vertices = intersections[sorted_indices]

# Σχεδίαση περιοχής με fill
#plt.fill(sorted_vertices[:, 0], sorted_vertices[:, 1], color='lightgreen', alpha=0.4, label='Εφικτή περιοχή')

# Με γραφικό τρόπο βρείτε  τη βέλτιστη κορυφή του προβλήματος, εάν υπάρχει. # Υπολογισμός Ζ
#slider
# Ζ = 3x1 + x2

z_values = [Z(x, y) for x, y in intersections]

# Δημιουργία γραφήματος
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

# Σχεδίαση περιορισμών
ax.plot(x, line1(x), label='0.3x1 + 0.1x2 <= 2.7')
# 
# 0.6x1 + 0.4x2 >=6
ax.plot(x, line2(x), label='0.5x1 + 0.5x2 = 6')
ax.plot(x, line3(x), label='0.6x1 + 0.4x2 >=6')
ax.plot(x, line5(x), label='x1 (x2 = 0)')
ax.plot(0 * x, x, label='x2 (x1 = 0)')
ax.axhline(0, color='black', lw=0.5)

# Σχεδίαση εφικτής περιοχής
ax.fill(sorted_vertices[:, 0], sorted_vertices[:, 1], color='lightgreen', alpha=0.4, label='Εφικτή περιοχή')

# Σχεδίαση κορυφών
for i, (xv, yv) in enumerate(intersections):
    ax.plot(xv, yv, 'ro')
    ax.text(xv + 0.1, yv, f'P{i+1}', fontsize=9)
    x_val, y_val = intersections[i]
    print(f'Κορυφή P{i+1}: {round(xv,3)}, {round(yv,3)}, Z:{round(Z(x_val, y_val), 3)} ')
# Προσθήκη ευθείας Z
initial_z = 2
z_line, = ax.plot(x, 2.*initial_z - 0.8 * x, 'r--', label='Z = 0.4 x₁ + 0.5 x₂')  #x2=2z-0.8x1

def currentvertex(z_val):
    z_val = z_slider.val
    tol=0.02
    for x, y in intersections:
        if abs(round(Z(x, y),3) - round(z_val,3)) <= tol:
            #print(f'Κορυφή: {x}, {y}')
            return np.array([x, y])
    return None

highlight, = ax.plot([], [], 'mo', markersize=10, label='Κοντινή κορυφή')

# Slider
ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
z_slider = Slider(ax_slider, 'Z', -10,  10, valinit=initial_z, valstep=0.0001, valfmt='%1.2f')

# Slider update function
def update(val):
    z_val = z_slider.val
    z_line.set_ydata(np.round(2. * z_val - 0.8 * x, 3))

    bv = currentvertex(z_val)
    if bv is not None:
        highlight.set_data([bv[0]], [bv[1]])
    else:
        highlight.set_data([], [])
    fig.canvas.draw_idle()


z_slider.on_changed(update)
update(initial_z)

ax.set_xlim(-2, 15)
ax.set_ylim(-2, 15)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Γραφική εύρεση βέλτιστης κορυφής με slider αντικειμενικής συνάρτησης Z')
ax.legend()
plt.grid(True)
plt.show()

