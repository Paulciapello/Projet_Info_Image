#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def create_Ray(O, D):
    ray = {'origin': np.array(O),
           'direction': np.array(D)}
    return ray


def create_sphere(P, r, i):
    sphere = {'type': 'sphere',
              'centre': np.array(P),
              'rayon': np.array(r),
              'index_sphere': int(i)}
    return sphere


def create_plane(P, n, i):
    plane = {'position': np.array(P),
             'vect_n': np.array(n),
             'index_plane': int(i)}
    return plane


def normalize(x):
    return x / np.linalg.norm(x)


def rayAt(ray, t):
    O = ray['origin']
    D = ray['direction']
    return O + t * D


def get_Normal(obj, M):
    # Remplissez ici 
    if obj['type'] == 'sphere':
        N = normalize(M-obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N
    

def intersect_Plane(ray, plane):
    P = plane['position']
    n = plane['vect_n']
    O = ray['origin']
    D = ray['direction']
    if abs(np.dot(P, n)) < 1e-6:
        return np.inf
    else:
        t = np.dot((P - O), n) / np.dot(D, n)
        if t > 0:
            return t
        else:
            return np.inf


def intersect_Sphere(ray, sphere):
    O = ray['origin']
    D = ray['direction']
    P = sphere['centre']
    r = sphere['rayon']

    a = np.dot(D, D)
    b = -2 * np.dot(D, (P - O))
    c = np.dot((P - O), (P - O)) - r**2

    delta = b**2 - 4 * a * c
    if delta >= 0:
        t1 = (-b - math.sqrt(delta)) / (2 * a)
        t2 = (-b + math.sqrt(delta)) / (2 * a)
        if t1 >= 0 and t2 >= 0:
            return min(t1, t2)
        else:
            return np.inf
    else:
        return np.inf

    
def intersect_Scene(ray, obj):
    if obj['type'] == 'plane':
        return intersect_Plane(ray,obj)
    elif obj['type'] == 'sphere':
        return intersect_Sphere(ray, sphere)
    
    

def Is_in_Shadow(obj_min,P,N):
    # Remplissez ici 
    return 


def eclairage(obj,light,P) : 
    # Remplissez ici 
    return 

def reflected_ray(dirRay,N):
    # Remplissez ici 
    return

def compute_reflection(rayTest,depth_max,col):
    # Remplissez ici 
    return col 

def trace_ray(ray):
    # Remplissez ici 
    return 


# Taille de l'image
w = 800
h = 600
acne_eps = 1e-4
materialShininess = 50


img = np.zeros((h, w, 3)) # image vide : que du noir
#Aspect ratio
r = float(w) / h
# coordonnées de l'écran : x0, y0, x1, y1.
S = (-1., -1. / r , 1., 1. / r )


# Position et couleur de la source lumineuse
Light = { 'position': np.array([5, 5, 0]),
          'ambient': np.array([0.05, 0.05, 0.05]),
          'diffuse': np.array([1, 1, 1]),
          'specular': np.array([1, 1, 1]) }

L = Light['position']


col = np.array([0.2, 0.2, 0.7])  # couleur de base
C = np.array([0., 0.1, 1.1])  # Coordonée du centre de la camera.
Q = np.array([0,0.3,0])  # Orientation de la caméra
img = np.zeros((h, w, 3)) # image vide : que du noir
materialShininess = 50
skyColor = np.array([0.321, 0.752, 0.850])
whiteColor = np.array([1,1,1])
depth_max = 10

scene = [create_sphere([.75, -.3, -1.], # Position
                         .6, # Rayon
                         #np.array([1. , 0.6, 0. ]), # ambiant
                         #np.array([1. , 0.6, 0. ]), # diffuse
                         #np.array([1, 1, 1]), # specular
                         #0.2, # reflection index
                         1), # index
          create_plane([0., -.9, 0.], # Position
                         [0, 1, 0], # Normal
                         #np.array([0.145, 0.584, 0.854]), # ambiant
                         #np.array([0.145, 0.584, 0.854]), # diffuse
                         #np.array([1, 1, 1]), # specular
                         #0.7, # reflection index
                         2), # index
         ]

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        #Remplisssez votre code ici
        img[h - j - 1, i, :] = np.clip(col, 0, 1) # la fonction clip permet de "forcer" col a être dans [0,1]

plt.imsave('figRaytracing.png', img)