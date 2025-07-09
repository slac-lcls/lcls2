import numpy as np

def CalcEnergy(m_amu,Px_au,Py_au,Pz_au):
    amu2au = 1836.15
    return 27.2*(Px_au**2 + Py_au**2 + Pz_au**2)/(2*amu2au*m_amu)
