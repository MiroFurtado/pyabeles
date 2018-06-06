'''
Simulates log(I) vs angle X-Ray Reflectivity curves for a given multilayered crystalline strucutre

Author: Miro Furtado
for the Hoffman Lab @ Harvard


TODO: Abstract away a Layer or Material class -> Doing, but currently no encapsulation of the Layer class so not the best practice. 
        * Fix Layer class encapsulation. 
      Get some clarity on units for Rho (is it different for Neutron vs X-Ray?)
      Solve the inverse problem
      Offload substrate simulation stuff onto the Experiment class -- ie. subst.doExperiment() returns an experiment class and then plotting it is part of Experiment.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Layer:

    def __init__(self, height, rho, irho=0., sigma=0., name = "", verbose=False):
        if verbose:
            print "Initializing new layer " + name
        self.name = name
        self.d = height
        self.rho = rho
        self.irho = irho
        self.sigma = sigma
        self.name = name

    def __add__(self, other):
        s = Surface(verbose=False)
        s.addLayer(self)
        s.addLayer(other)
        return s

    def __radd__(self,other):
        c = other.copy()
        c.addLayer(self)
        return c


class Scanner:
    ''' Class for the incident X-Ray probe, stores information about the wavelength (might extend as needed)
    '''
    def __init__(self,lbda=1.54056,xray=True, error=0.1): #Default value is Copper K alpha
        self.lbda = lbda #Units of 10^-10 m (Angstrom)
        self.xray = True
        self.error = error
    
    def getLambda(self):
        return self.lbda


class Surface:
    ''' Class for the multilayered structure, allows you to define an empty structure and add layers to it.
    '''
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.label = []
        self.N = 1
        self.d = np.array([0]) # Units of 10^-10 m (Angstrom) - Array of length N
        self.rho = np.array([0]) # Units of .... - Array of length N
        self.irho = np.array([0]) # Same units of above - Array of length N
        self.sigma = np.array([]) # Units of ... - Array of length N-1 (only care about junctions)
        if verbose:
            print("Initializing empty multilayered structure.")
        

    def doExperiment(self, angles, scanner=Scanner(), R=None, genR = True):
        return Experiment(angles, R=R, scanner=scanner, surf=self, genR = genR)


    def copy(self):
        c = Surface(verbose=False)
        c.verbose = self.verbose
        c.label = self.label
        c.N = self.N
        c.d = self.d
        c.rho = self.rho
        c.irho = self.irho
        c.sigma = self.sigma
        return c

    def getDistances(self):
        # Returns distances between different layers of the structure
        return self.d
    

    def addLayer(self,layer):
        if(layer.name != ""):
            name = " " + layer.name
        if(self.verbose):
            print("Adding new layer" + name + " to structure.")
        self.N += 1
        self.label.append(name)
        self.d = np.append(self.d, layer.d)
        self.rho = np.append(self.rho, layer.rho)
        self.irho = np.append(self.irho, layer.irho)
        self.sigma = np.append(self.sigma, layer.sigma)

    # DEPRECATED in favor of addLayer, this function allows you to manually add a new layer.
    def addLayer_d(self,dist, rhoN, irhoN=0, sigmaN=0, name = ""):
        print("WARNING: Deprecated in favor of addLayer")
        if(name != ""):
            name = " " + name
        if(self.verbose):
            print("Adding new layer" + name + " to structure.")

        self.N += 1
        self.label.append(name)
        self.d = np.append(self.d,dist)
        self.rho = np.append(self.rho,rhoN)
        self.irho = np.append(self.irho,irhoN)
        self.sigma = np.append(self.sigma,sigmaN)
    
    #My implementation of the Abeles matrix formalism, built off of 
    # https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)#Abeles_matrix_formalism
    def abeles(self, kz):
        if self.verbose:
            print("Iterating through Abeles transfer-matrix method.")
        k = kz

        def getK(n): #Generates the wavevector for a given interface
            return np.sqrt(kz**2-4e-6*np.pi*(self.rho[n]+1j*self.irho[n]-self.rho[0]-1j*self.irho[0]))
        def getR(n,k,kNext, nevotCroce=True): #Returns the Fresnel reflection coefficient between n and n+1
            Rn = (k-kNext)/(k+kNext)
            if(nevotCroce): #Use the Nevot and Croce correction for surface roughness. TODO: Allow for custom error func
                Rn *= np.exp(-2*k*kNext*self.sigma[n]**2)
            return Rn
        def getBeta(k, n):
            if n==0: return 0.
            else:
                return 1j*k*self.d[n]

        # Unfortunately, can't use numpy matrix libraries because we have an array of values
        # Resultant Matrix initialized as the Identity matrix
        M00 = 1
        M11 = 1
        M01 = 0
        M10 = 0
        
        for i in range(0,self.N-1): #Iterate through everything except the substrate
            kNext = getK(i+1) #Next wavenumber
            beta = getBeta(k,i) #Phase factor
            R = getR(i, k, kNext) #Fresnel coefficient
            
            #Characteristic matrix
            C00 = np.exp(beta)
            C11 = np.exp(-beta)
            C10 = R*np.exp(beta)
            C01 = R*np.exp(-beta)
            
            #Multiplying the matrix (T for temporary matrix)
            T00 = M00*C00 + M10*C01
            T10 = M00*C10 + M10*C11
            T01 = M01*C00 + M11*C01
            T11 = M01*C10 + M11*C11
            
            M00 = T00
            M11 = T11
            M01 = T01
            M10 = T10
            k = kNext
            
        r = M01/M00
        return np.absolute(r)**2


class Experiment:
    ''' Variables
    x in degrees is the list of incident angles measured in the experiment
    scanner is the object for the probe that includes information about the wavelength it scans at, etc. 
    surface is the multilayered structure that the experiment is being conducted on
    theory is the theoretical R curve (as opposed to the measured one with error)
    refl is the measured reflection values
    lRefl is the log10 of the measured reflection values
    '''
    def __init__(self, angles, R=None, scanner=Scanner(), surf=Surface(verbose=False), genR=True):
        self.x = angles
        self.scanner = scanner
        self.surface = surf
        self.verbose = self.surface.verbose
        self.theory = R
        if genR and R is None:
            R = self.genTheory(self.x, degrees=True)
            self.theory = R
        if R is not None:
            self.lRefl = np.log10(R)
            self.refl = R
        else:
            self.lRefl = None
            self.refl = None

    #Simulates the X-Ray reflection intensity curve for the material - Note if degrees are specified, it overrides the experiment. 
    def genTheory(self, thetas=None, degrees=True, modify=True):
        if thetas is not None and modify:
            self.x = thetas
        if(self.verbose):
            print("Simulating X-Ray reflection intensity curve.")
        if degrees:
            thetas = np.radians(self.x)
        lbda = self.scanner.getLambda()
        kz = 2*np.pi/lbda*np.sin(thetas)
        R = self.surface.abeles(kz)
        if modify:
            self.refl = R
            self.lRefl = np.log10(R)
        return R

    def simulateData(self, thetas=None, noise = 1., degrees=True, modify=True):
        if thetas is not None:
            refl = genTheory(thetas, modify=modify)
        else:
            refl = self.refl
        errorV = noise*0.1*self.refl
        #errorV[errorV==0] = 1e-11
        refl = refl + np.random.randn(len(self.refl))*errorV
        if modify:
            self.refl = refl
            self.lRefl = np.log10(refl)
        return refl

    def simulatePlot(self,thetas=None, degrees=True):
        plt.title("X-Ray Reflectivity Curve")
        plt.ylabel("log(Intensity)")
        plt.xlabel("Theta (degrees)")
        if thetas is not None:
            self.x = thetas
        plt.plot(self.x,np.log10(self.genTheory(thetas=thetas, degrees=degrees)))

    def resids(self): #Error seems to be a constant, so its inclusion in the minimization seems unnecessary.  
        if self.lRefl is None:
            print("Can't calculate residuals before simulating!")
        lRhat = np.log10(self.genTheory(self.x, modify=False))
        toRet = (self.lRefl-lRhat)**2/(self.scanner.error**2*np.ones(len(lRhat)))
        return toRet

    def nllf(self, d): #as a first go at it, let's try simulating where we know the surface almost completely EXCEPT for incorrect knowledge of gaps
        self.surface.d = d
        return 0.5*np.sum(self.resids()) # + a constant term from uncertainty we neglect because, uh, it's a constant.

    def fixSpacing(self, guess):
        res = minimize(self.nllf, guess, method='nelder-mead')
        self.surface.d = res.x
        if not res.success:
            print("Failed to converge to a correct structure.")
        return self.surface
