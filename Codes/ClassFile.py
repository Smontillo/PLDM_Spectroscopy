import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit
import Parameters as par
# ===================================

spec = [
    ('ind_FL',           int32[:, :]),
    ('ind_SL',           int32[:, :]),
    ('w_FL',           complex128[:]),
    ('θ_SL',           float64[:, :]),
    ('r_SL',           float64[:, :]),
    ('μψ',             complex128[:]),
    ('μψ2',             complex128[:,:]),
    ('ψF0',            complex128[:, :]),
    ('ψB0',            complex128[:, :]),
    ('NSteps',                 int32),
    ('Dir',                int32),
    ('ntj',                int32),
    ('nData',              int32),
    ('Laser',              int32),
    ('intiStateF',             int32),
    ('intiStateB',             int32),
    ('R',           complex128[:, :]),
    ('P',           complex128[:, :]),
    ('Rw',           complex128[:, :, :]),
    ('Pw',           complex128[:, :, :]),
    ('v',           complex128[:, :]),
    ('ψF',          complex128[:, :]),
    ('ψB',          complex128[:, :]),
    ('ψFw',         complex128[:, :, :]),
    ('ψBw',         complex128[:, :, :]),
    ('ρt',             complex128[:]),
    ('ρRe',         complex128[:, :]),
    ('Force1',      complex128[:, :]),
    ('Force2',      complex128[:, :])
]
@jitclass(spec)
class TrajData(object):
    def __init__(self, NSteps, ntj, nData):
        self.NSteps = NSteps
        self.Dir    = 0
        self.ntj    = ntj
        self.nData  = nData
        self.Laser  = 0
        self.ind_FL = np.zeros((self.ntj,2), dtype = np.int32)
        self.w_FL   = np.ones((self.ntj), dtype = np.complex128)
        self.ind_SL = np.zeros((self.ntj,2), dtype = np.int32)
        self.θ_SL   = np.zeros((self.ntj,2), dtype = np.float64)
        self.r_SL   = np.zeros((self.ntj,2), dtype = np.float64)
        self.μψ     = np.zeros((par.Nt), dtype = np.complex128)
        self.μψ2     = np.zeros((par.Nt,self.ntj), dtype = np.complex128)
        self.ψF0    = np.zeros((par.Nt, self.ntj), dtype = np.complex128)
        self.ψB0    = np.zeros((par.Nt, self.ntj), dtype = np.complex128)
        # BATH PARAMETERS
        self.R      = np.zeros((par.NSites * par.ndof, self.ntj), dtype = np.complex128)
        self.P      = np.zeros((par.NSites * par.ndof, self.ntj), dtype = np.complex128)
        self.Rw     = np.zeros((self.nData, par.NSites * par.ndof, self.ntj), dtype = np.complex128)
        self.Pw     = np.zeros((self.nData, par.NSites * par.ndof, self.ntj), dtype = np.complex128)
        self.v      = np.zeros((par.NSites * par.ndof, self.ntj), dtype = np.complex128)
        # ELECTRONIC PARAMETERS
        self.ψF     = np.zeros((par.Nt, self.ntj), dtype = np.complex128)
        self.ψB     = np.zeros((par.Nt, self.ntj), dtype = np.complex128)
        self.ψFw    = np.zeros((self.nData, par.Nt, self.ntj), dtype = np.complex128)
        self.ψBw    = np.zeros((self.nData, par.Nt, self.ntj), dtype = np.complex128)
        # POPULATION
        self.ρt     = np.zeros((par.Nt * par.Nt), dtype = np.complex128)
        self.ρRe    = np.zeros((self.nData, par.Nt * par.Nt), dtype = np.complex128)
        # FORCES
        self.Force1 = np.zeros((par.NSites * par.ndof, self.ntj), dtype = np.complex128)
        self.Force2 = np.zeros((par.NSites * par.ndof, self.ntj), dtype = np.complex128)