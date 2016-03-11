# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.

TODO:
-----

- Should have a separate function or option to compute 3rd order
  moments + asymmetry (which are not always needed)

- remove alpha calculation (which is only about (0,0), and make a get
  alpha function that does it from an arbitrary point given a
  pre-computed list of parameters

"""
import numpy as np
import itertools
import math

__all__ = [
    'intersect_nominal'
]


def intersect_nominal(hillas_parameters):
    if len(hillas_parameters)<2:
        return None

    print("here")
    hillas_pairs = itertools.combinations((hillas_parameters), 2)
    print (hillas_pairs)

    sum_x = 0
    sum_y = 0
    sum_w = 0

    for hill in hillas_pairs:
        h1,h2 = hill
        #print (h1,h2)
        x1,y1 = intersect_lines(h1.cen_x,h1.cen_y,h1.psi,h2.cen_x,h2.cen_y,h2.psi)
        print(h2.psi,h1.psi,h1.cen_x,h2.cen_x)
        print("pos",x1,y1,h1.size,h2.size)
        weight = weight_amplitude(h1.size,h2.size)
        sum_x+=x1 * weight
        sum_y+=y1 * weight
        sum_w+=weight

    print ((sum_x/sum_w),(sum_y/sum_w))

def intersect_lines(xp1,yp1,phi1,xp2,yp2,phi2):

    #/* Hesse normal form for line 1 */
    s1 = math.sin(phi1)
    c1 = math.cos(phi1)
    A1 = s1
    B1 = -1*c1
    C1 = yp1*c1 - xp1*s1

    #/* Hesse normal form for line 2 */
    s2 = math.sin(phi2)
    c2 = math.cos(phi2)
    A2 = s2
    B2 = -1*c2
    C2 = yp2*c2 - xp2*s2

   # print("hmm1",A1,B1,C1,yp1,xp1,c1,s1, yp1*c1 ,xp1*s1)
   # print("hmm2",A2,B2,C2,yp2,xp2,c2,s2, yp2*c2 ,xp2*s2)

    detAB = (A1*B2-A2*B1)
    detBC = (B1*C2-B2*C1)
    detCA = (C1*A2-C2*A1)

    if  math.fabs(detAB) < 1e-14 : # /* parallel */
        return 0,0

    xs = detBC / detAB
    ys = detCA / detAB

    return xs,ys


def weight_amplitude(amp1,amp2):
    return (amp1*amp2)/(amp1+amp2)
