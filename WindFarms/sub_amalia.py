import numpy as np
import matplotlib.pyplot as plt
import prettify

locations = np.genfromtxt('layout_amalia.txt', delimiter=' ')
tX = locations[:,0]
tY = locations[:,1]
diameter = 126.4  # meters  # This is for plotting the layouts to scale
colors = prettify.tableau_colors()

subgroup1 = [0, 1, 3, 4, 8, 9, 14, 15, 21]
subgroup2 = [34, 41, 49, 55, 58, 48, 54]
subgroup3 = [54, 55, 56, 57, 58, 59]
subgroup4 = [53, 46, 40, 33, 47]
subgroup5 = [6, 7, 11, 12, 13, 18, 19]

# Plot the layout
fig, ax = plt.subplots()
for i, (x, y, d) in enumerate(zip(tX, tY, [diameter]*len(tX))):
    ax.add_artist(plt.Circle((x, y), radius=d/2., color=colors[7]))
    ax.text(x, y, str(i))
ax.set_axis_off()
ax.set_xlim([-190, 3990])
ax.set_ylim([-245, 5145])
ax.set_aspect('equal')


# Plot the subgroups
subgroups = [subgroup1, subgroup2, subgroup3, subgroup4, subgroup5]
for subgroup in subgroups:
    tXsub = [tX[i] for i in subgroup]
    tYsub = [tY[i] for i in subgroup]

    fig, ax = plt.subplots()
    for i, (x, y, d) in enumerate(zip(tXsub, tYsub, [diameter]*len(tX))):
        ax.add_artist(plt.Circle((x, y), radius=d/2., color=colors[7]))
    ax.set_axis_off()
    ax.set_xlim([-190, 3990])
    ax.set_ylim([-245, 5145])
    ax.set_aspect('equal')

plt.show()

# Write out the sub layouts
# for j, subgroup in enumerate(subgroups, 1):
#     tXsub = [tX[i] for i in subgroup]
#     tYsub = [tY[i] for i in subgroup]
#     np.savetxt('layout_amalia_sub%d.txt' % j, np.c_[tXsub, tYsub], header='x, y')
