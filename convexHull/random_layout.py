"""
Description:    Generates random wind farm turbine layout within a given convex boundary
Date:           2/22/2017
Author:         Santiago Padron, modified from Jared J. Thomas
"""

from wakeexchange.GeneralWindFarmComponents import calculate_distance, calculate_boundary

import numpy as np
import matplotlib.pyplot as plt
from time import sleep


def generate_layout(locations, npoints=60):

    # get unit normals
    boundaryVertices, unit_normals = calculate_boundary(locations)
    print unit_normals

    # initialize points array
    points = np.zeros([npoints, 2])

    # deterine high and low values for containing rectangle
    turbineX = locations[:, 0]
    turbineY = locations[:, 1]
    highx = max(turbineX)
    lowx = min(turbineX)
    highy = max(turbineY)
    lowy = min(turbineY)

    print highx
    print lowx
    print highy
    print lowy

    # generate random points within the wind farm boundary
    np.random.seed(101)
    counter = 0
    for i in range(0, npoints):

        good_point = False

        while not good_point:

            # generate random point in containing rectangle
            point = np.random.rand(1, 2)
            # print point
            point[0, 0] = point[0, 0]*(highx - lowx) + lowx
            point[0, 1] = point[0, 1]*(highy - lowy) + lowy
            # print point

            # calculate signed distance from the point to each boundary face
            distance = calculate_distance(point, boundaryVertices, unit_normals)

            # determine if the point is inside the wind farm boundary
            if all(d >= 0 for d in distance[0]):
                good_point = True
                points[i, :] = point[0, :]
            counter += 1

    return points, boundaryVertices


if __name__ == '__main__':

    # Turbine locations
    locations = np.loadtxt('../WindFarms/layout_amalia.txt')
    print locations
    print len(locations)

    points, vertices = generate_layout(locations, npoints=60)
    pointsx = points[:, 0]
    pointsy = points[:, 1]

    np.savetxt('outputfiles/layout.txt', np.c_[pointsx, pointsy])

    # Plot
    # # plot points
    # plt.scatter(points[:, 0], points[:, 1])
    # # plot boundary
    # plt.plot(vertices[:, 0], vertices[:, 1], 'k')
    # plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], 'k')

    # Plot to scale
    diameter = 126.4  # m
    fig, ax = plt.subplots(figsize=(10, 10))
    for x, y in zip(pointsx, pointsy):
        ax.add_artist(plt.Circle((x, y), radius=diameter/2, color='b'))

    # # Range just to encircle the turbines
    ax.set_xlim([-190, 3990]) # Range of variation of the Amalia times +0.05% on each side.
    ax.set_ylim([-245, 5145])
    # ax.set_ylim([-1245, 5145])

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    # Get the boundary (the convex hull)
    locations = np.column_stack((pointsx, pointsy))

    # Plot the convex hull boundary
    # vertices, unused = calculate_boundary(locations)  # if you want the convex hull of the random layout.
    ax.plot(vertices[:, 0], vertices[:, 1], '--', color='grey')
    ax.plot(np.array([vertices[0, 0], vertices[-1, 0]]), np.array([vertices[0, 1], vertices[-1, 1]]), '--', color='grey')

    plt.show()
