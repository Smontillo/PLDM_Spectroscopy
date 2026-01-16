import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit
import Parameters as par
# ===================================

spec = [
    ('NSteps',        int32),
    ('intiStateF',    int32),
    ('intiStateB',    int32),
    ('R',             complex128[:, :]),
    ('P',             complex128[:, :]),
    ('v',             complex128[:, :]),
    ('ψF',            complex128[:, :]),
    ('ψB',            complex128[:, :]),
    ('ρt',               complex128[:]),
    ('ρRe',           complex128[:, :]),
    ('Force1',        complex128[:, :]),
    ('Force2',        complex128[:, :])
]
@jitclass(spec)
class TrajData(object):
    def __init__(self, NSteps):
        self.NSteps = NSteps
        self.intiStateF = 1
        self.intiStateB = 1
        # BATH PARAMETERS
        self.R      = np.zeros((par.NSites * par.ndof, par.ntj), dtype = np.complex128)
        self.P      = np.zeros((par.NSites * par.ndof, par.ntj), dtype = np.complex128)
        self.v      = np.zeros((par.NSites * par.ndof, par.ntj), dtype = np.complex128)
        # ELECTRONIC PARAMETERS
        self.ψF     = np.zeros((par.Nt, par.ntj), dtype = np.complex128)
        self.ψB     = np.zeros((par.Nt, par.ntj), dtype = np.complex128)
        # POPULATION
        self.ρt     = np.zeros((par.Nt * par.Nt), dtype = np.complex128)
        self.ρRe    = np.zeros((par.nData, par.Nt * par.Nt), dtype = np.complex128)
        # FORCES
        self.Force1 = np.zeros((par.NSites * par.ndof, par.ntj), dtype = np.complex128)
        self.Force2 = np.zeros((par.NSites * par.ndof, par.ntj), dtype = np.complex128)