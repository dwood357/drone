import numpy as np
import matplotlib.pyplot as plt

def rotation(phi, theta, psi):

	R = np.array([[np.cos(psi)*np.cos(theta), np.cos(theta)*np.sin(psi), -np.sin(theta)],
		[np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi), 
		np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta), np.cos(theta)*np.sin(phi)],
		[np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta),
		np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi), np.cos(phi)*np.cos(theta)]])
	print(R)
	return R

def rk4(f, y, dt):
    """
    Runge-Kutta 4th Order

    Solves an autonomous (time-invariant) differential equation of the form dy/dt = f(y).
    """
    k1 = f(y)
    k2 = f(y + dt/2 * k1)
    k3 = f(y + dt/2 * k2)
    k4 = f(y + dt   * k3)
    
    rk = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return rk

def position(x,y,z,steps):

	x = np.linspace(0,x)
	y = np.linspace(0,y)
	z = np.linspace(0,z)


class Quadrotor(object):
    """
    Quadrotor: Holds the dynamics of the system, initialized with the current state

    Output will be the control u.
       
    """
    def __init__(self, r=None, Phi=None, v=None, omega=None):
        
        # internal quadrotor state
        self.r = r if r is not None else np.zeros((3,1))
        self.Phi = Phi if Phi is not None else np.zeros((3,1))
        self.v = v if v is not None else np.zeros((3,1))
        self.omega = omega if omega is not None else np.zeros((3,1))
        
        # phyiscal true parameters
        self.g = 9.81
        self.mass = 3.81
        Ixx = 0.060224 
        Iyy = 0.122198
        Izz = 0.132166
        
        self.I = np.array([[Ixx,0,0],
                           [0,Iyy,0],
                           [0,0,Izz]])
        
        self.Mu = np.diag(np.array([0.85,0.85,0.85])) # drag
#         self.Mu = np.diag(np.array([0,0,0])) # drag
        
        # max control actuation
        self.Tmax = 40
        self.Mmax = 2
        
        # convenience
        self.Niters = 0
        
    def __str__(self):
        s  = "Quadrotor state after {} iters:\n".format(self.Niters)
        s += "\tr:     {}.T\n".format(self.r.T)
        s += "\tPhi:   {}.T\n".format(self.Phi.T)
        s += "\tv:     {}.T\n".format(self.v.T)
        s += "\tomega: {}.T\n".format(self.omega.T)
        return s
    
    @staticmethod
    def clamp(v, limit):
        v = np.copy(v)
        v[np.abs(v) > limit] = np.sign(v[np.abs(v) > limit])*limit
        return v

    @staticmethod
    def Gamma(phi, theta):
        gamma = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0,         np.cos(phi)      ,        -np.sin(phi)      ],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ], dtype = float)
        return gamma
        
    def Fg(self, phi, theta):
    	"""
    	Potential Energy due to gravity
    	"""
        Fg = self.mass*self.g*np.array([[-np.sin(theta)],
                                        [np.cos(theta)*np.sin(phi)],
                                        [np.cos(theta)*np.cos(phi)]])
        return Fg

    def Fr(self, F, phi, theta):
    	"""
		Force Resultant, hand it an array of Forces
    	"""
    	Fr = np.array([[-np.sin(theta)*np.sum(F)],
						[np.cos(theta)*np.sin(phi)*np.sum(F)],
                        [np.cos(theta)*np.cos(phi)*np.sum(F)]]) - Fg(phi,theta)

    	return Fr
    
    def update(self, u, dt):
        # We've been through another iteration
        self.Niters += 1
        
        # Extract Euler angles for convenience
        ph = self.Phi.flatten()[0]
        th = self.Phi.flatten()[1]
        ps = self.Phi.flatten()[2]
        
        #
        # Forces and Moments
        #
        
        u = np.array(u).flatten()
        T = np.array([[0, 0, u[0]]]).T
        M = np.array([[u[1], u[2], u[3]]]).T
        
        #
        # Saturate control effort
        #
        
        T = self.clamp(T, self.Tmax)
        M = self.clamp(M, self.Mmax)
        
        #
        # Kinematics
        #
        
        # Translational
#         f = lambda r: Rot_i_to_b(ph,th,ps).T.dot(self.v) # body velocities
        f = lambda r: self.v # inertial velocities
        self.r = rk4(f, self.r, dt)
        
        # Rotational
        f = lambda Phi: self.Gamma(Phi[0], Phi[1]).dot(self.omega)
        self.Phi = rk4(f, self.Phi, dt)
        
        #
        # Dynamics
        #
        
        # Translational
#         f = lambda v: (1/self.mass)*(self.Fg(ph,th) - T - np.cross(self.omega, v, axis=0)) # body
        f = lambda v: (1/self.mass)*(self.Fg(0,0) - rotation(ph,th,ps).T.dot(T) - self.Mu.dot(v)) # inertial
        # f = lambda v: (1/self.mass)*(self.Fg(0,0) - rotation(ph,th,ps).T.dot(T) - self.Mu.dot(v)) # inertial
        self.v = rk4(f, self.v, dt)
        
        # Rotational
        f = lambda omega: np.linalg.inv(self.I).dot((-np.cross(omega, self.I.dot(omega), axis=0) + M))
        self.omega = rk4(f, self.omega, dt)
        
        # update control input
        u = np.hstack((T[2], M.flatten()))
        return u


class SMC(Controller):
    """Sliding Mode Controller
    
    This implementation of SMC includes the ability
    to track reference signals, as opposed to the
    StabilizingSMC class above. In addition, this
    class utilizes a saturation function instead
    of the naive signum function for the switching
    component of the controller.
    """
    def __init__(self):
        self.name = "SMC"
        
        # dirty-derivative filters
        self.dyidt = DirtyDerivative(order=1, tau=0.0125)
        self.dyidt2 = DirtyDerivative(order=2, tau=0.025)
        self.dyodt = DirtyDerivative(order=1, tau=0.025)
        self.dyodt2 = DirtyDerivative(order=2, tau=0.05)
        
        # Value of total thrust from last iteration
        self.u1_d1 = 0
        
        # estimates of the physical properties of the quadrotor
        self.g = 9.81
        self.mass = 3.81
        self.I = np.diag(np.array([0.1, 0.1, 0.1]))
        
        # proportional control law \phi(\eta) that stabilizes the inner \dot\eta = f_a system
        self.K1 = np.diag(np.array([2,5,5,1]))
        
        # proportional control law \phi(\eta) that stabilizes the outer \dot\eta = f_a system
        self.K2 = np.diag(np.array([1,1]))
               
    def inner(self, desired, state, Ts):
        
        def fa(eta, xi):
            return xi

        def fb(eta, xi):
            # extract omega from xi
            omega = xi[1:4]

            r = np.zeros((4,1))
            r[0,0] = self.g
            r[1:4] = -np.linalg.inv(self.I).dot(np.cross(omega, self.I.dot(omega), axis=0))
            return r
        
        # Create the reference signal vector
        y = np.atleast_2d(desired).T # make a col vector
        ydot = self.dyidt.update(y, Ts)
        yddot = self.dyidt2.update(ydot, Ts)
        
        # Transform the system into the form of HK (14.4) and (14.5)
        eta = np.array([state.flatten()[[2, 6, 7, 8]]]).T # r_z, ph, th, ps
        xi = np.array([state.flatten()[[5, 9, 10, 11]]]).T # \dot{r_z}, p, q, r

        # for convenience
        ph = eta.flatten()[1]
        th = eta.flatten()[2]
        ps = eta.flatten()[3]
        
        # Build the inv(E(x)) matrix
        Einv = np.diag(np.array([-1,1,1,1]))
        
        # Build the L(x) = inv(G(x)) matrix
        L = np.eye(4)
        L[0, 0] = self.mass/(np.cos(ph)*np.cos(th))
        L[1:4, 1:4] = self.I
        
        # continuous control to cancel known terms: equation (18) of [1].
        ueq = -Einv.dot(L.dot( (fb(eta, xi) - yddot) + self.K1.dot( fa(eta-y, xi-ydot) )))
        
        # Build the matrix of sliding surfaces
        SS = (xi-ydot) + self.K1.dot(eta-y)
        
        # Gains
        Beta = np.diag(np.array([11,5,5,2]))
        
        # switching components of the control: equation (20) of [1]
        # (but with a boundary on the sliding surface to mitigate chattering)
        gamma = -Beta.dot(sat( SS/1 ))
        
        # equation (23) of [1]
        u = ueq + Einv.dot(gamma)
        
        return u
    
    def outer(self, desired, u1, state, Ts):
        
        # Make sure the thrust is non-zero
        if u1 == 0:
            return np.array([0,0])
        
        def fa(eta, xi):
            return xi

        def fb(eta, xi):
            return np.zeros((2,1))
        
        # Create the reference signal vector
        y = np.atleast_2d(desired).T # make a col vector
        ydot = self.dyodt.update(y, Ts)
        yddot = self.dyodt2.update(ydot, Ts)
                
        # Transform the system into the form of HK (14.4) and (14.5)
        eta = np.array([state.flatten()[[0, 1]]]).T # r_x, r_y
        xi = np.array([state.flatten()[[3, 4]]]).T # \dot{r}_x, \dot{r}_y
        
        ps = state.flatten()[8]
        
        # Build the inv(E(x)) matrix
        R = np.array([
            [ np.cos(ps), np.sin(ps)],
            [-np.sin(ps), np.cos(ps)],
        ])
        Einv = R.T.dot(np.diag(np.array([-1.0/u1,1.0/u1])))
        
        # Build the L(x) = inv(G(x)) matrix
        L = self.mass*np.eye(2)
        
        # continuous control to cancel known terms: equation (18) of [1].
        ueq = -Einv.dot(L.dot( (fb(eta, xi) - yddot) + self.K2.dot( fa(eta-y, xi-ydot) )))
        
        # Build the matrix of sliding surfaces
        SS = (xi-ydot) + self.K2.dot(eta-y)
        
        # Gains
        Beta = np.diag(np.array([15,15]))
        
        # switching components of the control: equation (20) of [1]
        # (but with a boundary on the sliding surface to mitigate chattering)
        gamma = -Beta.dot(sat(SS/1))
        
        # equation (23) of [1]
        u = ueq + Einv.dot(gamma)
        
        return u
    
    def update(self, commanded, state, pkt, Ts):
        
        #
        # Outer Loop
        #
        
        ref = self.outer(commanded.flatten()[[0, 1]], self.u1_d1, state, Ts)
        
        #
        # Inner Loop
        #
        
        # build reference signal for inner loop
        rz_ref = commanded.flatten()[2]
        ph_ref = ref.flatten()[1]
        th_ref = ref.flatten()[0]
        ps_ref = commanded.flatten()[8]
        desired = np.array([rz_ref, ph_ref, th_ref, ps_ref])
        
        u = self.inner(desired, state, Ts)

        # Save total thrust for the outer loop
        self.u1_d1 = u.flatten()[0]
        
        # update the commanded states
        commanded[6] = ph_ref
        commanded[7] = th_ref
        
        # actuator commands
        return u, commanded

