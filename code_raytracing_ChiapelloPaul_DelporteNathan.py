# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:50:24 2024

@author: chp4223a
"""

import numpy as np                       
import matplotlib.pyplot as plt          # Import des module python utile au projet


def create_ray(O, D):
    """Crée un dictionnaire rayon avec les clés suivantes:"""
    ray = {'origin': np.array(O),
           'direction': np.array(D)}
    return ray


def create_sphere(C, r, amb, dif, sp, ref, i):
    """Crée un dictionnaire sphere avec les clés suivantes:"""
    sphere = {'type': 'sphere',
              'centre': np.array(C),
              'rayon': np.array(r),
              'diffuse': np.array(dif),
              'ambient': np.array(amb),
              'specular': np.array(sp),
              'reflection': ref,
              'index': int(i)}
    return sphere


def create_plane(P, n, amb, dif, sp, ref, i):
    """Crée un dictionnaire plan avec les clés suivantes:"""
    plane = {'type': 'plane',
             'position': np.array(P),
             'vect_n': np.array(n),
             'diffuse': np.array(dif),
             'ambient': np.array(amb),
             'specular': np.array(sp),
             'reflection': ref,
             'index': int(i)}
    return plane


def create_cylinder(C, r, z, amb, dif, sp, ref, i):
    """Crée un dictionnaire cylindre avec les clés suivantes:"""
    cylinder = {'type': 'cylinder',
                'centre': np.array(C),
                'rayon': np.array(r),
                'height': np.array(z),
                'diffuse': np.array(dif),
                'ambient': np.array(amb),
                'specular': np.array(sp),
                'reflection': ref,
                'index': int(i)}
    return cylinder


def normalize(x):
    """Normalise un vecteur"""
    return x / np.linalg.norm(x)


def ray_at(ray, t):
    """Calul les points de la droite "rayon" suivant la variable t"""
    return ray['origin'] + t * ray['direction']


def get_normal(obj, M):
    """Calcul la normal d'un vecteur a un point donné M sur un objet."""
    if obj['type'] == 'sphere':
        Vect = normalize(M - obj['centre'])                                 # pour une sphere 
    elif obj['type'] == 'plane':
        Vect = obj['vect_n']                                                # pour un plan
    elif obj['type'] == 'cylinder':
        H = obj['centre']
        Vect = normalize(np.array([M[0] - H[0], M[1] - H[1], 0]))           # pour un cylindre 
    return Vect


def intersect_plane(ray, plane):
    """Calcul l'intersetion entre un rayon et un plan sinon renvoie un np.inf"""
    P = plane['position']
    n = plane['vect_n']
    O = ray['origin']
    d = ray['direction']
    if abs(np.dot(d, n)) < 1e-6:
        return np.inf
    t = -np.dot((O - P), n) / np.dot(d, n)
    if t > 0:
        return t
    return np.inf


def intersect_sphere(ray, sphere):
    """Calcul l'intersetion entre un rayon et une sphere sinon renvoie un np.inf"""
    O = ray['origin']
    d = ray['direction']
    C = sphere['centre']
    r = sphere['rayon']

    a = np.dot(d, d)
    b = -2 * np.dot(d, (C - O))                   #Coef de l'equation quadratique du cercle à qui on a injecter l'équation de la droite rayon
    c = np.dot((C - O), (C - O)) - r**2           #de la forme ax**2 + bx + c

    delta = b**2 - 4 * a * c
    if delta >= 0:
        t1 = (-b - np.sqrt(delta)) / (2 * a)            #Calcul des points t1,t2 d'intersection
        t2 = (-b + np.sqrt(delta)) / (2 * a)                    
        if t1 >= 0 and t2 >= 0:
            return min(t1, t2)
        else:
            return np.inf
    else:
        return np.inf


def intersect_cylinder(ray, cylinder):
    O = ray['origin']
    d = ray['direction']
    C = cylinder['centre']
    R = cylinder['rayon']
    z = 10
    
    x1 = O[0]
    y1 = O[1]       #récupere les (x1,y1) de O
    #O_z = O[2]
    
    a = d[0]
    b = d[1]       #récupere les (a,b) du vecteur direction
    #c = d[2]
    
    x0 = C[0]
    y0 = C[1]       #récupere les(xo,yo,zo) de C
    z0 = C[2]
    
    A = a**2 + b**2                                       #Coef de l'equation quadratique du cylindre à qui on a injecter l'équation de la droite rayon
    B = 2 * (x1 * a - x0 * a - y1 * b + y0 * b)                                            
    C = x1**2 - 2 * x1 * x0 + x0**2 + y1**2 - 2 * y1 * y0 + y0**2 - R**2 + (z - z0)**2
    
    delta = B**2 - 4 * A * C
    
    if delta >= 0:
        t_1 = (-B-np.sqrt(delta))/(2*A)
        t_2 = (-B+np.sqrt(delta))/(2*A)                                     #Calcul des points t1,t2 d'intersection
        if t_1 >= 0 and t_2 >= 0:
            return min(t_1, t_2)
        else:
            return np.inf
    else:
        return np.inf


def intersect_scene(ray, obj):
    """Renvoie les bonnes fonctions suivant l'objet intersecté"""
    if obj['type'] == 'plane':
        return intersect_plane(ray, obj)
    elif obj['type'] == 'sphere':
        return intersect_sphere(ray, obj)
    elif obj['type'] == 'cylinder':
        return intersect_cylinder(ray, obj)


def is_in_shadow(obj_min, P, N):
    """Verifie si les points intersecté sont dans l'ombre"""
    PL = normalize(L - P)                          # Calcul la direction de la lumière 
    ray_test = create_ray(P + acne_eps * N, PL)    # crée un rayon d'origine P + acne_eps * N dans la direction de la lumière        
    I_intersect = []                       # Listera tous les points intersecté

    for obj in scene:
        if obj['index'] != obj_min['index']:            
            t_obj = intersect_scene(ray_test, obj)
            if t_obj != np.inf:
                I_intersect.append(t_obj)

    if I_intersect:         
        return True              
    return False


def eclairage(obj, Light, P):
    """Calcul l'éclairage à un point donné sur un objet."""
    N = get_normal(obj, P)
    PL = normalize(L - P)             
    PC = normalize(C - P)

    # Calcul la couleur en fonction des modèles utilisés
    ct = obj['ambient'] * Light['ambient']

    # Illumination de Lambert (diffuse)
    ct += obj['diffuse'] * Light['diffuse'] * max(np.dot(N, PL), 0)

    # Illumination de Blinn-Phong (specular)
    ct += obj['specular'] * Light['specular'] * max(np.dot(N, normalize(PL + PC)), 0)**materialShininess
    return ct


def reflected_ray(dir_ray, N):
    """Calcul la direction du rayon réfléchi."""
    return dir_ray - 2 * np.dot(dir_ray, N) * N


def compute_reflection(ray_test, depth_max, col):
    """Calcul la couleur de la réfelction"""
    d = ray_test['direction']
    c = 1

    for k in range(1, depth_max):
        traced = trace_ray(ray_test)

        if traced is None:
            break

        obj, M, N, col_ray = traced
        col = col + c * col_ray
        d = reflected_ray(ray_test['direction'], N)
        ray_test = create_ray(M + acne_eps * N, d)
        c = c * obj['reflection']

    return col


def trace_ray(ray):
    """Trace un rayon à travers la scene"""
    t_min = np.inf
    obj_min = None

    for obj in scene:
        t_obj = intersect_scene(ray, obj)
        if t_obj < t_min:
            t_min, obj_min = t_obj, obj

    if obj_min is None:
        return None

    P = ray_at(ray, t_min)
    N = get_normal(obj_min, P)
    shadow = is_in_shadow(obj_min, P, N)

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

for u in range(0,1000):     # Calcul des 1000 images 
    
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
                             0.4, # reflection index
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
            rayTest = create_ray(C, D)
            traced = trace_ray(rayTest)
            if traced :
                obj, M, N, col_ray = traced
                col = compute_reflection(rayTest,depth_max,col)
            img[h - j - 1, i, :] = np.clip(col, 0, 1) # la fonction clip permet de "forcer" col a être dans [0,1]
    
    plt.imsave('figRaytracing'+str(u)+'.png', img)
    u = u + 1






































