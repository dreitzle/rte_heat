# coding: utf-8

"""heat3L_det.py: Explicit 3-layered solution determinants"""

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"


# Robin boundaries determinants

def calc_M_3L_delta(k1, k2, k3, a1, a2, a3, e1, e2, e3, p1, p2, p3,
                     m1, m2, m3, h1, h2):
    g1 = a2*k2*p2*(a3*p3+h2*m3)
    g2 = a3*k3*m2*(a3*m3+h2*p3)
    g3 = a2*k2*m2*(a3*p3+h2*m3)
    g4 = a3*k3*p2*(a3*m3+h2*p3)
    g5 = a2*k2*(a3*p3+h2*m3)*(h1*k1*p2+a2*k2*m2)
    g6 = a3*k3*(a3*m3+h2*p3)*(h1*k1*m2+a2*k2*p2)
    return a1*a1*k1*m1*(g1+g2)+a2*k2*h1*m1*(g3+g4)+a1*p1*(g5+g6)

# Src in layer 1

def calc_A_3L_delta_q1_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a2*k2*(a3*p3+h2*m3)*(a1*k1*p2-a2*k2*m2)
    t2 = a3*k3*(a3*m3+h2*p3)*(a1*k1*m2-a2*k2*p2)
    return (e1*f1+(1.0+h1/a1)*f2)*(t1+t2)/k1


def calc_B_3L_delta_q1_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a1*(f1+e1*f2)-h1*e1*f2
    t2 = a1*(f1-e1*f2)+h1*e1*f2
    t3 = a1*k1*p2*t1+a2*k2*m2*t2
    t4 = a1*k1*m2*t1+a2*k2*p2*t2
    return (a2*k2*(a3*p3+h2*m3)*t3 + a3*k3*(a3*m3+h2*p3)*t4)/(a1*k1)


def calc_A_3L_delta_q1_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a2*k2*(a3*p3+h2*m3)
    t2 = a3*k3*(a3*m3+h2*p3)
    return 2.0*e2*(a1*(e1*f1+f2)+h1*f2)*(t1-t2)


def calc_B_3L_delta_q1_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a2*k2*(a3*p3+h2*m3)
    t2 = a3*k3*(a3*m3+h2*p3)
    return 2.0*(a1*(e1*f1+f2)+h1*f2)*(t1+t2)


def calc_A_3L_delta_q1_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return 4.0*a2*k2*e2*e3*(a3-h2)*(a1*(e1*f1+f2)+h1*f2)


def calc_B_3L_delta_q1_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return 4.0*a2*k2*e2*(a3+h2)*(a1*(e1*f1+f2)+h1*f2)

# Src in layer 2

def calc_A_3L_delta_q2_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a3*k3*(f1-e2*f2)*(h2*k3*p3+a3*k3*m3)
    t2 = a2*k2*(f1+e2*f2)*(h2*k3*m3+a3*k3*p3)
    return 2.0/k3*(h1+a1)*(t1+t2)

def calc_B_3L_delta_q2_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a3*k3*(f1-e2*f2)*(h2*k3*p3+a3*k3*m3)
    t2 = a2*k2*(f1+e2*f2)*(h2*k3*m3+a3*k3*p3)
    return 2.0*e1/k3*(a1-h1)*(t1+t2)

def calc_A_3L_delta_q2_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = k1*a1*(f1*e2-f2)*(h1*k1*p1+a1*k1*m1)
    t2 = k2*a2*(f1*e2+f2)*(h1*k1*m1+a1*k1*p1)
    t3 = a2*k2*(h2*k3*m3+a3*k3*p3)
    t4 = a3*k3*(h2*k3*p3+a3*k3*m3)
    return (-t1+t2)*(t3-t4)/(a2*k1*k2*k3)

def calc_B_3L_delta_q2_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = -a1*a1*k1*k1*m1+a2*k2*h1*k1*m1-a1*k1*p1*(h1*k1-a2*k2)
    t2 = a3*k3*(f1-e2*f2)*(h2*k3*p3+a3*k3*m3)
    t3 = a2*k2*(f1+e2*f2)*(h2*k3*m3+a3*k3*p3)
    return t1*(t2+t3)/(a2*k1*k2*k3)

def calc_A_3L_delta_q2_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a1*k1*(f1*e2-f2)*(h1*k1*p1+a1*k1*m1)
    t2 = a2*k2*(f1*e2+f2)*(h1*k1*m1+a1*k1*p1)
    return 2.0*e3/k1*(h2-a3)*(t1-t2)

def calc_B_3L_delta_q2_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a1*k1*(f1*e2-f2)*(h1*k1*p1+a1*k1*m1)
    t2 = a2*k2*(f1*e2+f2)*(h1*k1*m1+a1*k1*p1)
    return 2.0/k1*(h2+a3)*(-t1+t2)

# Src in layer 3

def calc_A_3L_delta_q3_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return calc_B_3L_delta_q1_z3(k3, k2, k1, a3, a2, a1, e3, e2, e1,
                                  p3, p2, p1, m3, m2, m1, h2, h1, f2, f1)

def calc_B_3L_delta_q3_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return calc_A_3L_delta_q1_z3(k3, k2, k1, a3, a2, a1, e3, e2, e1,
                                  p3, p2, p1, m3, m2, m1, h2, h1, f2, f1)

def calc_A_3L_delta_q3_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return calc_B_3L_delta_q1_z2(k3, k2, k1, a3, a2, a1, e3, e2, e1,
                                  p3, p2, p1, m3, m2, m1, h2, h1, f2, f1)

def calc_B_3L_delta_q3_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return calc_A_3L_delta_q1_z2(k3, k2, k1, a3, a2, a1, e3, e2, e1,
                                  p3, p2, p1, m3, m2, m1, h2, h1, f2, f1)

def calc_A_3L_delta_q3_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return calc_B_3L_delta_q1_z1(k3, k2, k1, a3, a2, a1, e3, e2, e1,
                                  p3, p2, p1, m3, m2, m1, h2, h1, f2, f1)

def calc_B_3L_delta_q3_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return calc_A_3L_delta_q1_z1(k3, k2, k1, a3, a2, a1, e3, e2, e1,
                                  p3, p2, p1, m3, m2, m1, h2, h1, f2, f1)
