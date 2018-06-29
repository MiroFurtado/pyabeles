'''
Simulates log(I) vs angle X-Ray Reflectivity curves for a given multilayered crystalline strucutre

Author: Miro Furtado
for the Hoffman Lab @ Harvard


TODO(Miro)
        * Fix Layer class encapsulation. 
      Explore ML for X-Ray reflectivity fitting -> Potentially big performance enhancements, running a NN is way faster than doing a search through a large parameter space.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping, differential_evolution as de
from scipy import stats

def beamfrac(FWHM, length, angle):
    """
    Calculate the beam fraction intercepted by a sample.
    Parameters

    Used under license: https://raw.githubusercontent.com/refnx/refnx/ca6e457412dcda1285d42644e0451f24b6b3c21c/LICENSE
    ----------
    FWHM: float
        The FWHM of the beam height
    length: float
        Length of the sample in mm
    angle: float
        Angle that the sample makes w.r.t the beam (degrees)
    Returns
    -------
    beamfrac: float
        The fraction of the beam that intercepts the sample
    """
    height_of_sample = length * np.sin(np.radians(angle))
    beam_sd = FWHM / 2 / np.sqrt(2 * np.log(2))
    probability = 2. * (stats.norm.cdf(height_of_sample / 2. / beam_sd) - 0.5)
    return probability

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
    
    def set_thickness(self,d):
        self.d = d
        return self

class Scanner:
    """ Class for the incident X-Ray probe, stores information about the wavelength (might extend as needed)
    """
    def __init__(self,lbda=1.54056,xray=True, error=0.1, background=0., offset=0.,beam_width_sd=0.019449): #Default value is Copper K alpha
        """
        Initializes a new Scanner object

        Parameters
        ----------
        offset: float
            Offset in measured theta

        """
        self.offset = offset #Theta offset in degrees
        self.lbda = lbda #Units of 10^-10 m (Angstrom)
        self.xray = True
        self.error = error
        self.width_sd = beam_width_sd #UNIT IS mm BE CAREFUL!
        self.background = background #Background level of detection, assumed to be constant.
    
    def getLambda(self):
        return self.lbda


class Surface:
    ''' Class for the multilayered structure, allows you to define an empty structure and add layers to it.
    '''
    def __init__(self, verbose=False, sample_width=1.):
        self.verbose = verbose
        self.width = sample_width #Units of mm (TODO: Double check this) - float
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
    
    #My implementation of the Abeles matrix formalism, built off of 
    # https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)#Abeles_matrix_formalism
    def abeles(self, kz):
        if self.verbose:
            print("Iterating through Abeles transfer-matrix method.")
        k = kz

        def getK(n): #Generates the wavevector for a given interface
            return np.sqrt(kz**2-4e-6*np.pi*(self.rho[n]+1j*self.irho[n]-self.rho[0]-1j*self.irho[0]))
        def getR(n,k,kNext, error='NC'): #Returns the Fresnel reflection coefficient between n and n+1
            Rn = (k-kNext)/(k+kNext)
            if(error is 'NC'): #Use the Nevot and Croce correction for surface roughness. TODO: Allow for custom error func
                Rn *= np.exp(-2*k*kNext*self.sigma[n]**2)
            elif(error is 'DW'): #Use Debye-Waller factor
                Rn *= np.exp(-2*k**2*self.sigma[n]**2)
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
            
        r = np.absolute(M01/M00)**2
        r /= r.max()

        return r


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
        self.kz = None
        self.scale = 1.
        if genR and R is None:
            R = self.genTheory(self.x, degrees=True)
            self.theory = R
        if R is not None:
            self.lRefl = np.log10(R)
            self.refl = R
        else:
            self.lRefl = None
            self.refl = None

    #Gets the parameters of the experiment as a list, useful for optimization. 
    def get_params_list(self):
        N = self.surface.N
        params = [self.scale,self.scanner.background, self.scanner.offset, self.surface.width]
        params.extend(self.surface.d[1:N-1])
        params.extend(self.surface.rho[1:])
        params.extend(self.surface.sigma)
        return params

    #Simulates the X-Ray reflection intensity curve for the material - Note if degrees are specified, it overrides the experiment. 
    def genTheory(self, thetas=None, degrees=True, modify=True):
        if thetas is not None and modify:
            self.x = thetas
        if(self.verbose):
            print("Simulating X-Ray reflection intensity curve.")
        footprint_correction = 1.
        if self.surface.width is not None:
            footprint_correction = beamfrac(self.scanner.width_sd *
                                                2.35,
                                                self.surface.width,
                                                self.x+self.scanner.offset)
        if degrees:
            thetas = np.radians(self.x+self.scanner.offset)
        lbda = self.scanner.getLambda()
        kz = 2*np.pi/lbda*np.sin(thetas)
        R = self.surface.abeles(kz)/footprint_correction
        R += self.scanner.background
        R /= R[0]
        if modify:
            self.theory = R
            self.kz = kz
        return R

    def simulateData(self, thetas=None, noise = 1., degrees=True, modify=True):
        if thetas is not None:
            refl = self.genTheory(thetas, modify=modify)
        else:
            refl = self.theory
        errorV = noise*0.1*self.refl
        #errorV[errorV==0] = 1e-11
        refl = refl + np.random.randn(len(self.refl))*errorV
        if modify:
            self.refl = refl
            self.lRefl = np.log10(refl) 
        return refl

    def theoryPlot(self,thetas=None, degrees=True):
        plt.title("X-Ray Reflectivity Curve")
        plt.ylabel("log(Intensity)")
        plt.xlabel("Theta (degrees)")
        if thetas is not None:
            self.x = thetas
        plt.plot(self.x,np.log10(self.genTheory(thetas=thetas, degrees=degrees)))

    def resids(self): #Residual between the measured/simulated data and the theoretical curve generated from the structure associated with the experiment. 
        if self.lRefl is None:
            print("Can't calculate residuals before simulating!")
        lRhat = np.log10(self.genTheory(self.x, modify=False))
        toRet = (np.log10(self.theory)-lRhat)**2/(self.scanner.error**2*np.ones(len(lRhat)))
        return toRet

class Fitter:
    def __init__(self, exp, method="nm", modify_default=True,cutoff_begin=0,cutoff_end=9223372036854775807, bounds = None):
        """
        Initializes a Fitter object for fitting X-Ray reflectivity curves.

        Parameters
        ----------
        exp : Experiment
            The Experiment object that we are trying to fit the curve for. This is important because it contains information about the sample, the scanning probe, etc.
        method : str
            String indicating what method of optimization you would like to use. Options include "nm" for Nelder-Mead simplex, "bh" for basinhopping/simiulated annealing.
        modify_default : bool
            Indicates whether the fit should modify the values in the Experiment object. Currently shakily implemented so no real guarantees.
        cutoff_begin : int
            Starting cutoff for what thetas to include in the fit (this can be useful to exclude non-linear beam footpring effects that can't be captured by the default beam footpring adjustments)
        cutoff_end : int
            Ending cutoff for what thetas to include in the fit (this can be useful to exclude background noise from the detector)
        """
        self.method = method
        self.cutoff_b=cutoff_begin
        self.cutoff_e=cutoff_end
        self.modify = modify_default
        self.exp = exp
        self.num_vars = 9 #TODO: This is unacceptable, but is kept here for testing purposes. Also, var name is not PEP8 compliant.
        self.fixed = np.full(self.num_vars, False) #TODO: Fix to include all variables, this only works by luck currently.

        self.bounds = bounds
        self.params_dict = {
            "background": 1,
            "theta offset": 2,
            "sample width": 3,
        }
    
    def bounds_from_guess(self,guess=None):
        """Initializes default values for the bounds from a guess, given that bounds is None

        Parameters
        ----------
        guess : np.array
            Array of guesses for the parameters.
        """
        if guess is None:
            guess = self.default_guess()
        self.bounds = [(0.75*a,1.25*a) for a in guess]
        self.bounds[1] = (0.,2*guess[1])
        self.bounds[2] = (-2*guess[2],2*guess[2])
        self.bounds[3] = (0.1,5)

    
    #Fixes the num-th variable.
    def set_fixed(self, num):
        """Sets the num-th variable to be a fixed variable unchanged by the fit.

        Parameters
        ----------
        num : int
            Value of the variable to be fixed.
        """
        self.fixed[num] = True
    
    def set_free(self, num):
        """Sets the num-th variable to be a free variable

        Parameters
        ----------
        num : int
            Value of the variable to be freed.
        """
        self.fixed[num] = False

    def set_bound(self, num, bound):
        self.bounds[num] = bound

    def default_guess(self):
        """Returns a default guess
        """
        guess = [1.,1e-6]
        guess.extend([0.1,1.])
        guess.extend(self.exp.surface.d[1:self.exp.surface.N-1])
        guess.extend(self.exp.surface.rho[1:])
        guess.extend(self.exp.surface.sigma)
        guess = np.array(guess)
        return guess

    def fit(self, guess=None):
        """
        Fits the given data

        Parameters
        ----------
        guess : np.array
            Starting parameters to use when minimizing. Note: Starting guess is only used by simplex and basinhopping methods. DiffEv does not use guess. 
        """
        exp = self.exp
        if guess is None:
            guess = self.default_guess()
        if(self.method is "nm"):
            res = minimize(self.error, guess, method='nelder-mead',options={'maxiter':10000})
        elif(self.method is "bh"):
            res = basinhopping(self.error, guess, niter=1000)
            return exp.surface, res.x
        elif(self.method is "de"):
            if self.bounds is None:
                self.bounds_from_guess(guess)
            res = de(self.error, self.bounds,popsize=20,mutation=(1.,1.5))
        if not res.success:
            print("Failed to converge to a correct structure.")
            return exp.surface, res.x
        else:
            #exp.surface.d = res.x
            return exp.surface, res.x

    def error(self, guess):
        # Parameter space is the entirety-> Sigma, d, etc. 
        # Parameter vector p: first element is the background radiation, next N elements are thicknesses, next N elements are densities, next N-1 elements are sigmas
        N = self.exp.surface.N
        j = 4
        if self.fixed is not None:
            params = self.exp.get_params_list()
            for i in range(0, len(self.fixed)):
                if(self.fixed[i]):
                    guess[i] = params[i]
        self.exp.scale = guess[0] #Currently this is unused
        self.exp.scanner.background = guess[1]
        self.exp.scanner.offset = guess[2]
        self.exp.surface.width = guess[3]
        self.exp.surface.d[1:N-1] = guess[j:N+j-2]
        self.exp.surface.rho[1:] = guess[N+j-2:2*N+j-3]
        self.exp.surface.sigma = guess[2*N+j-3:]
        return np.sum(self.exp.resids()[self.cutoff_b:self.cutoff_e])

