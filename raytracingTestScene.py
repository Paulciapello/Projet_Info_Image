#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def create_Ray(O, D):
    ray = {'origin': np.array(O),
           'direction': np.array(D)}
    return ray


def create_sphere(C, r, amb, dif, sp, ref, i):
    sphere = {'type':'sphere',                                                                  
              'centre': np.array(C),
              'rayon': np.array(r),
              'diffuse': np.array(dif),
              'ambient' : np.array(amb),
              'specular': np.array(sp),
              'reflection': ref,
              'index' : int(i)}
    return sphere

def create_plane(P, n, amb, dif, sp,ref, i):
    plane = {'type' : 'plane',
             'position': np.array(P),
             'vect_n': np.array(n),
             'diffuse': np.array(dif),
             'ambient' : np.array(amb),
             'specular': np.array(sp),
             'reflection': ref,
             'index' : int(i)}
    return plane

def create_cylinder(C, r, z, amb, dif, sp, ref, i):
    cylinder = {'type':'cylinder',
                'centre':np.array(C),
                'rayon': np.array(r),
                'height': np.array(z),
                'diffuse': np.array(dif),
                'ambient' : np.array(amb),
                'specular': np.array(sp),
                'reflection': ref,
                'index' : int(i)}
    return cylinder

def normalize(x):
    return x/np.linalg.norm(x)


def rayAt(ray, t):
    return ray['origin']+t*ray['direction']


def get_Normal(obj, M):
    if obj['type'] == 'sphere':
        Vect =normalize(M-obj['centre'])
    elif obj['type'] == 'plane':
        Vect = obj['vect_n']
    elif obj['type'] == 'cylinder':
        H = obj['centre']
        
        Vect = normalize(M[0]-H[0],M[1]-H[1],0)
        
    return Vect
    

def intersect_Plane(ray, plane):
    P = plane['position']
    n = plane['vect_n']
    O = ray['origin']
    d = ray['direction']
    if abs(np.dot(d, n)) < 1e-6:
        return np.inf
    t = -np.dot((O-P),n)/np.dot(d, n)
    if t>0:
        return t
    return np.inf

def intersect_Sphere(ray, sphere):
    O = ray['origin']
    d = ray['direction']
    C = sphere['centre']
    r = sphere['rayon']

    a = np.dot(d, d)
    b = -2 * np.dot(d, (C-O))
    c = np.dot((C-O),(C-O)) - r**2

    delta = b**2 - 4*a*c
    if delta >= 0:
        t1 = (-b-np.sqrt(delta))/(2*a)
        t2 = (-b+np.sqrt(delta))/(2*a)
        if t1 >= 0 and t2 >= 0:
            return min(t1, t2)
        else:
            return np.inf
    else:
        return np.inf

def intersect_Cylinder(ray, cylinder):
    O = ray['origin']
    d = ray['direction']
    C = cylinder['centre']
    R = cylinder['rayon']
    z = 10
    
    x1 = O[0]
    y1 = O[1]       #récupere les (x1,y1,z1) de O
    #O_z = O[2]
    
    a = d[0]
    b = d[1]       #récupere les (a,b,c) du vecteur direction
    #c = d[2]
    
    x0 = C[0]
    y0 = C[1]       #récupere les(xo,yo,zo) de C
    z0 = C[2]
    
    A = a**2 + b**2
    B = 2 * (x1 * a - x0 * a - y1 * b + y0 * b)
    C = x1**2 - 2 * x1 * x0 + x0**2 + y1**2 - 2 * y1 * y0 + y0**2 - R**2 + (z - z0)**2
    
    delta = B**2 - 4 * A * C
    
    if delta >= 0:
        t_1 = (-B-np.sqrt(delta))/(2*A)
        t_2 = (-B+np.sqrt(delta))/(2*A)
        if t_1 >= 0 and t_2 >= 0:
            return min(t_1, t_2)
        else:
            return np.inf
    else:
        return np.inf
    
def intersect_Scene(ray, obj):
    if obj['type'] == 'plane':
        return intersect_Plane(ray, obj)
    elif obj['type'] == 'sphere':
        return intersect_Sphere(ray, obj)
    elif obj['type'] == 'cylinder':
        return intersect_Cylinder(ray, obj)
    
    

def Is_in_Shadow(obj_min,P,N):
    PL = normalize(L-P)
    rayTest = create_Ray(P+acne_eps*N,PL)
    I_intersect = []
    for obj in scene :
        if obj['index'] != obj_min['index']:
            t_obj = intersect_Scene(rayTest, obj)
            if t_obj != np.inf:
                I_intersect.append(t_obj)
    if I_intersect : 
        return True
    return False


def eclairage(obj,Light,P): 
    N = get_Normal(obj, P)
    PL = normalize(L-P)
    PC = normalize(C-P)
#on calcule la couleur suivant les modèles utilisé
    ct = obj['ambient']*Light['ambient']
#Lambert shading (diffuse).
    ct += obj['diffuse']*Light['diffuse']*max(np.dot(N,PL),0)
#Blinn-Phong shading (specular).
    ct += obj['specular']*Light['specular']*max(np.dot(N,normalize(PL+PC)),0)**(materialShininess)
    return ct
   

def reflected_ray(dirRay,N):
   return dirRay - 2*np.dot(dirRay,N)*N

def compute_reflection(rayTest,depth_max,col):
    d = rayTest['direction']
    c = 1
    for k in range(1,depth_max):
        traced = trace_ray(rayTest)      #### A MODIFIER
        if traced==None:
            break
        obj, M, N, col_ray = traced
        col = col + c*col_ray
        d = reflected_ray(rayTest['direction'],N)
        rayTest = create_Ray(M+acne_eps*N, d)
        c = c*obj['reflection']
    return col

def trace_ray(ray):
    t_min = np.inf
    obj_min = None
    for obj in scene :
        t_obj=intersect_Scene(ray, obj)
        if t_obj<t_min:
            t_min, obj_min =t_obj, obj
    if obj_min==None:
        return None
    P = rayAt(ray,t_min)
    N = get_Normal(obj_min,P)
    shadow = Is_in_Shadow(obj_min,P,N)
    if shadow:
        return None    
    col_ray = eclairage(obj_min, Light, P)
    return obj_min, P, N, col_ray





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

for u in range(0,1000):
    
    a = -1.2
    b = -1.6
    c = 0.1
    x = u/40 - 1.5
    z = -u/40
    y = np.abs(np.exp(-c * z) * a * np.cos((2 * np.pi / b) * z))-0.3
   
    scene = [create_sphere([x, y, z], # Position
                             .6, # Rayon
                             np.array([1. , 0.6, 0. ]), # ambiant
                             np.array([1. , 0.6, 0. ]), # diffuse
                             np.array([1, 1, 1]), # specular
                             0.2, # reflection index
                             1), # index
              create_plane([0., -.9, 0.], # Position
                             [0, 1, 0], # Normal
                             np.array([0.145, 0.584, 0.854]), # ambiant
                             np.array([0.145, 0.584, 0.854]), # diffuse
                             np.array([1, 1, 1]), # specular
                             0.7, # reflection index
                             2),# index
              create_cylinder([.75, -.3, -1.], 10.,3., np.array([1. , 0.6, 0. ]),
                              np.array([1. , 0.6, 0. ]), np.array([1, 1, 1]), 0.2, 3)
             ]
    
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        if i % 10 == 0:
            print(i / float(w) * 100, "%")
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            col = np.zeros((3))
            Q[:2] = (x,y)
            D = normalize(Q-C)
            rayTest = create_Ray(C, D)
            traced = trace_ray(rayTest)
            if traced :
                obj, M, N, col_ray = traced
                col = compute_reflection(rayTest,depth_max,col)
            img[h - j - 1, i, :] = np.clip(col, 0, 1) # la fonction clip permet de "forcer" col a être dans [0,1]
    
    plt.imsave('figRaytracing'+str(u)+'.png', img)
    u = u + 1
    