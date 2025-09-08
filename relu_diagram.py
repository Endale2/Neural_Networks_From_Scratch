# relu_diagram.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

plt.rcParams.update({'font.size': 12, 'figure.facecolor': '#071228'})

fig, ax = plt.subplots(figsize=(9.8,4.6))
ax.set_xlim(0,100); ax.set_ylim(0,50)
ax.axis('off')

# positions
x_in = 10
x_hidden = 50
x_out = 90
y_in = [35, 15]
y_hidden = [32, 18]
y_out = [25]

# draw neurons
def draw_neuron(x,y,label,small=None, r=3.5):
    circ = patches.Circle((x,y), radius=r, facecolor='#062033', edgecolor='#4a657a', linewidth=1.5)
    ax.add_patch(circ)
    ax.text(x, y, label, ha='center', va='center', color='white', fontsize=10)
    if small is not None:
        ax.text(x, y - r - 3, small, ha='center', va='top', color='#9fb2d9', fontsize=9)

# inputs
draw_neuron(x_in, y_in[0], 'x1', '1')
draw_neuron(x_in, y_in[1], 'x2', '2')

# hidden (A,B)
draw_neuron(x_hidden, y_hidden[0], 'A', 'z=-1\\nReLU→0', r=6)
draw_neuron(x_hidden+18, y_hidden[1], 'B', 'z=3\\nReLU→3', r=6)

# output
draw_neuron(x_out, y_out[0], 'y', '3', r=7)

# wires: lines list [(x1,y1),(x2,y2)]
lines = [
    [(x_in+4, y_in[0]), (x_hidden-6, y_hidden[0])],  # x1->A
    [(x_in+4, y_in[1]), (x_hidden-6, y_hidden[0])],  # x2->A
    [(x_in+4, y_in[0]), (x_hidden+12, y_hidden[1])], # x1->B
    [(x_in+4, y_in[1]), (x_hidden+12, y_hidden[1])], # x2->B
    [(x_hidden+6, y_hidden[0]), (x_out-6, y_out[0])],# A->out
    [(x_hidden+22, y_hidden[1]), (x_out-6, y_out[0])] # B->out
]

lc = LineCollection(lines, linewidths=[1.2,1.0,1.8,1.0,2.2,2.2], colors=['#9fb2d9','#9fb2d9','#7c3aed','#9fb2d9','#cfe7ff','#cfe7ff'])
ax.add_collection(lc)

# weights labels
ax.text(30, 38, 'w=1', color='#a9d0ff', fontsize=10)
ax.text(36, 22, 'w=-1', color='#a9d0ff', fontsize=10)
ax.text(60, 38, 'w=2', color='#a9d0ff', fontsize=10)
ax.text(62, 24, 'w=1', color='#a9d0ff', fontsize=10)
ax.text(75, 36, 'v1=1', color='#a9d0ff', fontsize=10)
ax.text(80, 26, 'v2=1', color='#a9d0ff', fontsize=10)

# title/legend
ax.text(5, 48, 'Tiny ReLU network (2 → 2 → 1)', color='#cfe7ff', fontsize=14)
ax.text(5, 45, 'Inputs: x1=1, x2=2   |   Target y=2', color='#94a3b8', fontsize=10)

plt.tight_layout()
plt.show()
