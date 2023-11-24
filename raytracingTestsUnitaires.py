#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import numpy.testing as nptest

w = 800
h = 600

img = np.zeros((h, w, 3))  # image vide : que du noir

C = np.array([0., 0.1, 1.1])  # Coordonnée du centre de la camera.
Q = np.array([0, 0.3, 0])  # Orientation de la caméra
img = np.zeros((h, w, 3))  # image vide : que du noir

r = float(w) / h
# coordonnées de l'écran : x0, y0, x1, y1.
S = (-1., -1. / r, 1., 1. / r)


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

       
#Test unitaire de la fonction codage. Si un test est faux, le message d'erreur vous dira quel test ne passe pas 
def test_traceRay():
    r1 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.08461529, -0.09804421, -0.99157833]))
    r2 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.03070244, -0.01463715, -0.99942139]))
    r3 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.0890529 , -0.10247237, -0.99074164]))
    decimal=8
    nptest.assert_array_almost_equal(rayAt(r1, 0.7821 ), np.array([0.06617762, 0.02331962, 0.32448659]), decimal) 
    print("r1 at t1 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r1, 0.1698 ), np.array([0.01436768, 0.08335209, 0.93163]), decimal)
    print("r1 at t2 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r1, -0.2345 ), np.array([-0.01984229,  0.12299137,  1.33252512]), decimal) 
    print("r1 at t3 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r2, 0.3581 ), np.array([0.01099454, 0.09475844, 0.7421072 ]), decimal) 
    print("r2 at t4 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r2, 0.0089 ), np.array([2.73251716e-04, 9.98697294e-02, 1.09110515e+00]), decimal)
    print("r2 at t5 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r2, -0.8345 ), np.array([-0.02562119,  0.1122147,   1.93401715]) , decimal) 
    print("r2 at t6 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r3, 0.5789 ), np.array([0.05155272, 0.04067875, 0.52645966]), decimal) 
    print("r3 at t7 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r3, 0.04521 ), np.array([0.00402608, 0.09536722, 1.05520857]), decimal)
    print("r3 at t8 : [OK]")
    nptest.assert_array_almost_equal(rayAt(r3, -0.0178 ), np.array([-0.00158514,  0.10182401,  1.1176352 ]), decimal) 
    print("r3 at t9 : [OK]")


def test_intersection_Plane():
    r1 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.08461529, -0.09804421, -0.99157833]))
    r2 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.03070244, -0.01463715, -0.99942139]))
    r3 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.0890529 , -0.10247237, -0.99074164]))
    p1 = create_plane([0.,-.7, -6.2],[0, 0, -1], 4)
    p2 = create_plane([0., -.9, 0.],[0, 1, 0], 5) 
    p3 = create_plane([-5, -.5, 0.], [1, 0, 0], 6)
    decimal=8
    nptest.assert_array_almost_equal(intersect_Plane(r1,p1),7.362000337381315, decimal,  err_msg="test intersect_Plane(r1,p1) Failed")
    print("r1 to plane1 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r1,p2),10.199480418068543,decimal,err_msg="test intersect_Plane(r1,p2) Failed")
    print("r1 to plane2 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r1,p3),np.inf,decimal, err_msg="test intersect_Plane(r1,p3) Failed")
    print("r1 to plane3 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r2,p1),7.304226298378506,decimal, err_msg="test intersect_Plane(r2,p1) Failed")
    print("r2 to plane1 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r2,p2),68.3193107947927, decimal,err_msg="test intersect_Plane(r2,p2) Failed")
    print("r2 to plane2 : [OK]")
    nptest.assert_equal(intersect_Plane(r2,p3),np.inf, err_msg="test intersect_Plane(r2,p3) Failed")
    print("r2 to plane3 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r3,p1),7.368217611202857,decimal, err_msg="test intersect_Plane(r3,p1) Failed")
    print("r3 to plane1 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r3,p2),9.75872813325192, decimal,err_msg="test intersect_Plane(r3,p2) Failed")
    print("r3 to plane2 : [OK]")
    nptest.assert_array_almost_equal(intersect_Plane(r3,p3),np.inf,decimal, err_msg="test intersect_Plane(r3,p3) Failed")    
    print("r3 to plane3 : [OK]")


def test_intersection_Sphere():
    r1 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.08461529, -0.09804421, -0.99157833]))
    r2 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.03070244, -0.01463715, -0.99942139]))
    r3 = create_Ray(np.array([0., 0.1, 1.1]), np.array([ 0.0890529 , -0.10247237, -0.99074164]))
    s1 = create_sphere([.75, -.3, -1.], .6, 1)
    s2 = create_sphere([.5, .1, -2.25], .4, 2)
    s3 = create_sphere([-.15, -.1, -.5],.3, 3)
    decimal=8
    nptest.assert_array_almost_equal(intersect_Sphere(r1,s1),2.143796176516055, decimal,err_msg="test intersect_Sphere(r1,s1) Failed")
    print("r1 to sphere1 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r1,s2),3.296010395154645,decimal, err_msg="test intersect_Sphere(r1,s2) Failed")
    print("r1 to sphere2 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r1,s3),1.5124664715654814,decimal, err_msg="test intersect_Sphere(r1,s3) Failed")
    print("r1 to sphere3 : [OK]")
    nptest.assert_equal(intersect_Sphere(r2,s1), np.inf,err_msg="test intersect_Sphere(r2,s1) Failed")
    print("r2 to sphere1 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r2,s2),3.356618090823908, decimal,err_msg="test intersect_Sphere(r2,s2) Failed")
    print("r2 to sphere2 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r2,s3),1.458922827460283, decimal,err_msg="test intersect_Sphere(r2,s3) Failed")
    print("r2 to sphere3 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r3,s1),2.060606939285078, decimal,err_msg="test intersect_Sphere(r3,s1) Failed")
    print("r3 to sphere1 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r3,s2),3.3369433177470023,decimal, err_msg="test intersect_Sphere(r3,s2) Failed")
    print("r3 to sphere2 : [OK]")
    nptest.assert_array_almost_equal(intersect_Sphere(r3,s3),1.5376144003766914,decimal, err_msg="test intersect_Sphere(r3,s3) Failed")
    print("r3 to sphere3 : [OK]")


print("Début des tests unitaires : ")
test_traceRay()
print("--> tests de la fonction traceRay : OK")
test_intersection_Plane()
print("--> tests de la fonction intersection_Plane : OK")
test_intersection_Sphere()
print("--> tests de la fonction intersection_Sphere : OK")