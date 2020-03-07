import numpy as np
import pandas as pd
import math
import csv
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from random import random
from collections import OrderedDict


vertices = np.array([(900, 350), (888, 855), (853, 855), (850, 779)])
vertices = np.array([(890, 350), (892, 855), (853, 855), (850, 779)])
# codes    = np.array([Path.MOVETO] + [Path.LINETO]* 3 + [Path.STOP])

roomsDimension = pd.read_csv('./01_data/room.csv')
doorsDimension = pd.read_csv('./01_data/door.csv')
# routesDimension = pd.read_csv('./01_data/route.csv')
routesDimension = pd.read_csv('./01_data/route2.csv')

plt.figure()
ax = plt.gca()

xlim_offset = 200
ylim_offset = 200
offset = 200
ax.axes.set_xlim([roomsDimension.min().values[0]-offset,roomsDimension.max().values[0]+offset])
ax.axes.set_ylim([roomsDimension.min().values[1]-offset,roomsDimension.max().values[1]+offset])

# currentAxis.add_patch(Rectangle((i[0],  i[1]), i[2],  i[3],alpha=.2,facecolor='yellow',edgecolor='none'))

def draw_rectangle(rectangle_dimensions, alpha=1, facecolor='none', edgecolor='black'):
    for i in rectangle_dimensions.values:
        ax.add_patch(Rectangle((i[0], i[1]), i[2], i[3], alpha=alpha, facecolor=facecolor, edgecolor=edgecolor))

def draw_map():
    draw_rectangle(roomsDimension)
    draw_rectangle(doorsDimension,edgecolor='red')
    draw_rectangle(routesDimension,alpha=0.3, edgecolor='none',facecolor='yellow')


def draw_walking_path(vertices, stepDistance, plot=plt):
    angleCoefficiences = np.array([])
    distances  = np.array([])
    dividers   = np.array([])
    xRange = np.array([])
    yRange = np.array([])

    for i in range(len(vertices)-1):
        k = (float(vertices[i+1][1]) - vertices[i][1]) / (vertices[i+1][0] - vertices[i][0])
        angleCoefficiences = np.append(angleCoefficiences, k)
        # print(angleCoefficiences)
        d = abs(math.sqrt(1+k**2) * (vertices[i+1][0] - vertices[i][0]))
        distances = np.append(distances, d)
        # print(distances)
        dd = math.floor(float(d)/stepDistance)
        dividers = np.append(dividers, dd)
        # print(dividers)
        xr = np.linspace(vertices[i][0],vertices[i+1][0],dd)
        xRange = np.append(xRange, xr)
        # print(xRange)
        yr = np.linspace(vertices[i][1],vertices[i+1][1],dd)
        yRange = np.append(yRange, yr)
        # print(yRange)
    # margin controls how "noisy" you want your fit to be.
    margin = 0.11
    noise  = margin*(np.random.random(len(yRange))-0.5)
    yRange = yRange + noise
    xRange = xRange + noise
    # ax = plot.gca()
    ax.plot(xRange, yRange, marker="o", ls="", ms=.7)

    # generate_walkingPath_xy(xRange,yRange)
    generate_walkingPath_errorCircle(xRange, yRange, plt)


    return xRange, yRange


def generate_walkingPath_xy(xRange, yRange):
    routeX, routeY = xRange, yRange
    nonRouteX, nonRouteY = xRange ,yRange
    indexArray = np.array([])
    notIndexArray = np.array([])

    for i, x in enumerate(nonRouteX):
        if(isOnRoute(routesDimension, nonRouteX[i], nonRouteY[i])):
            notIndexArray = np.append(notIndexArray, i)
        else:
            indexArray = np.append(indexArray, i)
    nonRouteX = np.delete(nonRouteX, notIndexArray)
    nonRouteY = np.delete(nonRouteY, notIndexArray)
    routeX    = np.delete(routeX, indexArray)
    routeY    = np.delete(routeY, indexArray)
    # print len(xRange), len(yRange)
    # print len(routeX), len(routeY)
    # print len(nonRouteX), len(nonRouteY)


    route    = pd.DataFrame({'x': routeX,    'y': routeY,    'label':  1})
    nonRoute = pd.DataFrame({'x': nonRouteX, 'y': nonRouteY, 'label':  0})
    data = pd.concat([route,nonRoute])
    data.to_csv('./01_data/walkingPathData.csv', index=False)


def generate_walkingPath_errorCircle(xRange, yRange, plot):
    routeX, routeY = xRange, yRange


    for i, x in enumerate(routeX):
        circle1 = plot.Circle((routeX[i], routeY[i]), 1.5, color='r')
        circle2 = plot.Circle((routeX[i], routeY[i]), 3,   color='b', alpha =0.2)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        # circle1 = plt.Circle((0, 0), 0.2, color='r')
        # circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
        # circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)



def unbalanced(sampleDensity, roomsDimension, routesDimension):
    routeX = np.array([])
    routeY = np.array([])
    nonRouteX = np.array([])
    nonRouteY = np.array([])
    for i in roomsDimension.values:
        area = i[2] * i[3]
        numberOfSamples = int(sampleDensity * area)
        # print(numberOfSamples)
        randomX = np.random.uniform(i[0],i[0]+i[2],numberOfSamples)
        randomY = np.random.uniform(i[1],i[1]+i[3],numberOfSamples)
        # print(len(randomX))
        for i in range(0,len(randomX)):
            if(isOnRoute(routesDimension, randomX[i], randomY[i])):
                # print('x: ' + str(randomX[i]) + '   y: ' + str(randomY[i]))
                routeX = np.append(routeX, randomX[i])
                routeY = np.append(routeY, randomY[i])
            else:
                nonRouteX = np.append(nonRouteX, randomX[i])
                nonRouteY = np.append(nonRouteY, randomY[i])

        ax = plot.gca()
        ax.plot(nonRouteX, nonRouteY, marker="o", color='g', ls="", ms=.7)
        ax.plot(routeX   , routeY   , marker="o", color='b', ls="", ms=.7)
        # plt.show()
    route    = pd.DataFrame({'x': routeX,    'y': routeY,    'label':  1})
    # print route
    nonRoute = pd.DataFrame({'x': nonRouteX, 'y': nonRouteY, 'label':  -1})
    # print nonRoute
    data = pd.concat([route,nonRoute])
    # print data
    # data.to_csv('./01_data/nnData_label_1_-1.csv', index=False)


def isOnRoute(routesDimension, xCoordinate, yCoordinate):
    isOnRoute = np.array([], dtype=bool)
    for i in routesDimension.values:
        if(xCoordinate>=i[0] and xCoordinate<=(i[0]+i[2]) and yCoordinate>=i[1] and yCoordinate<=(i[1]+i[3])):
            isOnRoute = np.append(isOnRoute, True)
        else:
            isOnRoute = np.append(isOnRoute, False)
        finalCheck = False
        for i in isOnRoute:
            finalCheck = (finalCheck or i)
    return finalCheck

def calculate_total_area(dimension):
    totalArea = 0
    for i in dimension.values:
        totalArea +=  i[2] * i[3]
    return totalArea

def draw_points_xy(xPoints,yPoints, marker="o", color='b', ls="", ms=.7):
    ax.plot(xPoints, yPoints, marker=marker, color=color, ls=ls,  ms=ms)


def get_random_xy(dimension, totalNumberOfSamples):
    randomX = np.array([])
    randomY = np.array([])
    for i in dimension.values:
        area = float(i[2] * i[3])
        # print ('area: ' + str(area))
        ratio = area/calculate_total_area(dimension)
        # print ('ratio: ' + str(ratio))
        numberOfSamples = int(ratio*totalNumberOfSamples)
        # print(' ,numberOfSample: '+str(numberOfSamples))
        randomX = np.append(randomX, np.random.uniform(i[0], i[0] + i[2], numberOfSamples))
        randomY = np.append(randomY, np.random.uniform(i[1], i[1] + i[3], numberOfSamples))
        # draw_points_xy(randomX,randomY)
    return randomX, randomY

def generate_route_xy(numberOfSamples):
    routeX, routeY = get_random_xy(routesDimension, numberOfSamples)
    return routeX, routeY


def generate_nonRoute_xy(numberOfSamples):
    nonRouteX, nonRouteY = get_random_xy(roomsDimension, numberOfSamples)
    indexArray = np.array([])

    for i, x in enumerate(nonRouteX):
        if(isOnRoute(routesDimension, nonRouteX[i], nonRouteY[i])):
            indexArray = np.append(indexArray, i)
    nonRouteX = np.delete(nonRouteX, indexArray)
    nonRouteY = np.delete(nonRouteY, indexArray)
    return nonRouteX, nonRouteY

def generate_map_xy(totalNumberOfSamples):
    roomArea  = calculate_total_area(roomsDimension)
    routeArea = calculate_total_area(routesDimension)
    ratio     = float(routeArea)/roomArea
    routeSamples      = int(totalNumberOfSamples/2.0)
    nonRouteSamples   = int((1/(2.0*(1-ratio)))*totalNumberOfSamples)

    rx, ry  = generate_route_xy(routeSamples)
    nrx,nry = generate_nonRoute_xy(nonRouteSamples)

    draw_points_xy(rx,ry,color='blue')
    draw_points_xy(nrx,nry,color='green')

    route    = pd.DataFrame({'x': rx,    'y': ry,    'label':  1})
    nonRoute = pd.DataFrame({'x': nrx,   'y': nry,   'label':  0})
    data = pd.concat([route,nonRoute])
    # print data
    # print(len(data))
    data.to_csv('./01_data/bRoute2Data_10k.csv', index=False)

def draw_data(filePath='./01_data/bRoute2Data_10k.csv'):
    # data = pd.read_csv(filePath).values
    data = pd.read_csv(filePath)
    routeData = data.loc[data['label']==1].values
    nonRouteData = data.loc[data['label']==0].values
    # print routeData
    rx, ry  = routeData[:, 1], routeData[:, 2]
    nrx,nry = nonRouteData[:, 1], nonRouteData[:, 2]

    draw_points_xy(rx,ry,color='blue')
    draw_points_xy(nrx,nry,color='green')

# generate_map_xy(10000)
draw_map()
draw_walking_path(vertices,7,plt)
# draw_data('./shuffledRoute1.csv')
# draw_data('./01_data/bnnData.csv')
# draw_data('./01_data/bRoute2Data_10k.csv')


# draw_data('./01_data/walkingPathData.csv')
plt.show()