import numpy as np
import matplotlib.pyplot as plt

"""
https://nbviewer.jupyter.org/github/plusk01/nonlinearquad/blob/master/sliding_mode_control.ipynb
"""

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def dynamics():
    f = np.array([[X7],
                [X8],
                [X9],
                [X10],
                [X11],
                [X12],
                [((Ix-Iz)/Iz)*(dq5**2)*np.cos(q5) + 
                ((Ix - Iy + Iz)/Iz)*dq4*dq5*np.cos(q5)*np.sin(q5) + 
                ((Ix - Iy + Iz)/Iz)*dq5*dq6*np.cos(q5) + 
                ((2*(Ix**2) + Iz**2 - 3*Ix*Iz)/(Ix*Iz))],
                [0],
                [0],
                [0],
                dq4*dq6*np.cos(q5),
                [0]])

    g = np.array([[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]
                [0,0,0,0]
                [0,0,0,0],
                [0,0,0,0],
                [g1, g1, g1, g1],
                [g2, g2, g2, g2],
                [g3, g3, g3, g3],
                [-(a/Ix)*np.sin(q5)*np.cos(q6), 0, (a/Ix)*np.sin(q5)*np.cos(q6), 0],
                [0, (a/Iq)*np.cos(q6), 0, -(a/Iq)*np.cos(q6)]])

def sat(v):
    v = np.copy(v)
    v[np.abs(v) > 1] = np.sign(v[np.abs(v) > 1])
    return v

def rk4(f, y, dt):
    """Runge-Kutta 4th Order

    Solves an autonomous (time-invariant) differential equation of the form dy/dt = f(y).
    """
    k1 = f(y)
    k2 = f(y + dt/2*k1)
    k3 = f(y + dt/2*k2)
    k4 = f(y + dt  *k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def rotation(phi, theta, psi):

    R = np.array([[np.cos(psi)*np.cos(theta), np.cos(theta)*np.sin(psi), -np.sin(theta)],
        [np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi), 
        np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta), np.cos(theta)*np.sin(phi)],
        [np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta),
        np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi), np.cos(phi)*np.cos(theta)]])
    return R

class Controller(object):
    """Controller
    """
    def __init__(self):
        self.name = "Manual"
    
    def __str__(self):
        return self.name
    
    def update(self, commanded, state, pkt, Ts):
        return np.array([[10, 0, 0, 0]]).T, commanded

class Estimator(object):
    """Estimator
    """
    def __init__(self, body=False):
        self.name = "Truth"
        
        # should the translational velocity be expressed
        # in the body-fixed frame or the inertial frame?
        self.body = body
        
    def __str__(self):
        return self.name
    
    def get_truth(self, quad):
        """Get Truth
        
        Obtain the true state from the simulated
        quadrotor object.
        """
        state = np.zeros((12,1))
        state[0:3] = quad.r
        # state[3:6] = Rot_i_to_b(*quad.Phi.flatten()).dot(quad.v) if self.body else quad.v
        state[3:6] = rotation(*quad.Phi.flatten()).dot(quad.v) if self.body else quad.v
        state[6:9] = quad.Phi
        state[9:12] = quad.omega
        return state
    
    def update(self, quad, u, Ts):
        """Update
        
        The default estimator is to use truth.
        """
        state = self.get_truth(quad)
        return state, state

class Commander(object):
    """Commander
    
    Allows the user to create flexible vehicle commands
    for the given controller. The default commander is
    a set point at 0.
    
    This class also allows the user to command any subset
    of vehicle states.
    """
    def __init__(self, default=False):
        # which subset of states should be commanded?
        self.pos = None
        self.vel = None
        self.Phi = None
        self.omega = None
        
        #
        # Default: Command everything to zero
        #
        
        if default:
            self.position(np.array([0.0, 0.0, 0.0]))
            self.velocity(np.array([None, None, None]))
            self.attitude(np.array([0.0, 0.0, 0.0]))
            self.angular_rates(np.array([None, None, None]))
        
    def position(self, pos):
        self.pos = pos
        
    def velocity(self, vel):
        self.vel = vel
        
    def attitude(self, Phi):
        self.Phi = Phi
        
    def angular_rates(self, omega):
        self.omega = omega
        
    def get_commands(self, i, Ts):
        pos = self.pos(i, Ts) if callable(self.pos) else self.pos
        Phi = self.Phi(i, Ts) if callable(self.Phi) else self.Phi
        commanded = np.hstack((pos, self.vel, Phi, self.omega))
        return commanded

class Simulator(object):
    """Simulator
        
    This class deals with the high-level simulation plumbing for a quadrotor system
    """
    def __init__(self, quad=None, ctrl=None, estm=None, cmdr=None, sens=None):
        self.quad = quad if quad else Quadrotor()
        self.ctrl = ctrl if ctrl else Controller()
        self.estm = estm if estm else Estimator()
        self.cmdr = cmdr if cmdr else Commander(default=True)
        self.sens = sens if sens else SensorManager()
        
        # Keep a history for plotting
        self.hist = {}
        
        # Simulation parameters
        self.Tf = 0
        self.Ts = 0
        self.N = 0
        
    def __str__(self):
        s  = "Simulation"
        return s

    def run(self, Tf, Ts=0.01):
        
        try:
            if _NO_SIM == True:
                return
        except Exception as e:
            pass
        
        # save simulation parameters
        self.Tf = Tf
        self.Ts = Ts
        
        # How many iterations are needed
        self.N = int(Tf/Ts)
        
        # quadrotor state
        state = truth = np.zeros((12,1))
        truth[0:3] = self.quad.r
        truth[3:6] = self.quad.v
        truth[6:9] = self.quad.Phi
        truth[9:12] = self.quad.omega
        
        # initialize data packet
        pkt = {}
        
        # initialize the plot history
        self.hist['u'] = np.zeros((4,self.N))
        self.hist['commanded'] = np.zeros((12,self.N))
        self.hist['state'] = np.zeros((12,self.N))
        self.hist['truth'] = np.zeros((12,self.N))
        
        #
        # Main Simulation Loop
        #
        
        for i in range(self.N):
            # determine the desired command
            commanded = self.cmdr.get_commands(i, Ts)
            
            # calculate control
            u, commanded = self.ctrl.update(commanded, truth, pkt, Ts)
                      
            # actuate physical model
            usat = self.quad.update(u, Ts)
            
            # read sensors
            pkt = self.sens.get_data_packet(self.quad, i, Ts)
            
            # run estimator
            state, truth = self.estm.update(self.quad, usat, Ts)
                
            # Update history
            self.hist['u'][:, i] = usat.flatten()
            self.hist['commanded'][:, i] = commanded.flatten()
            self.hist['state'][:, i] = state.flatten()
            self.hist['truth'][:, i] = truth.flatten()
    
    def plot(self):
        """Plot
        Create plot(s) of the evolution of the quadrotor state
        over the simulation horizon.
        """
        # Make sure that there is even data to plot
        if 'state' not in self.hist:
            return
        
        plt.ioff()
        fig = plt.figure(figsize=(12,10))
        fig.subplots_adjust(wspace=0.25)
        fig.suptitle('Vehicle State', fontsize=16)
        
        tvec = np.arange(self.N)*self.Ts
        
        #
        # Position
        #
        
        # for convenience
        xpos = self.hist['truth'][0, :]
        ypos = self.hist['truth'][1, :]
        zpos = self.hist['truth'][2, :]
        
        xcmd = self.hist['commanded'][0, :]
        ycmd = self.hist['commanded'][1, :]
        zcmd = self.hist['commanded'][2, :]
        
        ax = fig.add_subplot(6,2,1)
        if not np.isnan(xcmd).any():
            ax.plot(tvec, xcmd, 'r-', label='command')
        ax.plot(tvec, xpos, 'b-', label='truth')
        ax.set_ylabel('x'); ax.grid()
        ax.legend()

        ax = fig.add_subplot(6,2,3)
        if not np.isnan(ycmd).any():
            ax.plot(tvec, ycmd, 'r-', label='command')
        ax.plot(tvec, ypos, 'b-', label='truth')
        ax.set_ylabel('y'); ax.grid()
        
        ax = fig.add_subplot(6,2,5)
        if not np.isnan(zcmd).any():
            ax.plot(tvec, zcmd, 'r-', label='command')
        ax.plot(tvec, zpos, 'b-', label='truth')
        ax.set_ylabel('z'); ax.grid()
        
        #
        # Velocity
        #
        
        # for convenience
        xvel = self.hist['truth'][3, :]
        yvel = self.hist['truth'][4, :]
        zvel = self.hist['truth'][5, :]
        
        xvelhat = self.hist['state'][3, :]
        yvelhat = self.hist['state'][4, :]
        zvelhat = self.hist['state'][5, :]
        
        xcmd = self.hist['commanded'][3, :]
        ycmd = self.hist['commanded'][4, :]
        zcmd = self.hist['commanded'][5, :]
        
        ax = fig.add_subplot(6,2,2)
        if not np.isnan(xcmd).any():
            ax.plot(tvec, xcmd, 'r-', label='command')
        ax.plot(tvec, xvel, 'b-', label='truth')
        if not np.isnan(xvelhat).any() and self.estm.name != 'Truth':
            ax.plot(tvec, xvelhat, 'k:', label='estimate', linewidth=2)
        ax.set_ylabel('vx'); ax.grid()

        ax = fig.add_subplot(6,2,4)
        if not np.isnan(ycmd).any():
            ax.plot(tvec, ycmd, 'r-', label='command')
        ax.plot(tvec, yvel, 'b-', label='truth')
        if not np.isnan(yvelhat).any() and self.estm.name != 'Truth':
            ax.plot(tvec, yvelhat, 'k:', label='estimate', linewidth=2)
        ax.set_ylabel('vy'); ax.grid()
        
        ax = fig.add_subplot(6,2,6)
        if not np.isnan(zcmd).any():
            ax.plot(tvec, zcmd, 'r-', label='command')
        ax.plot(tvec, zvel, 'b-', label='truth')
        if not np.isnan(zvelhat).any() and self.estm.name != 'Truth':
            ax.plot(tvec, zvelhat, 'k:', label='estimate', linewidth=2)
        ax.set_ylabel('vz'); ax.grid()
        
        #
        # Attitude
        #
        
        # for convenience
        ph = self.hist['truth'][6, :]
        th = self.hist['truth'][7, :]
        ps = self.hist['truth'][8, :]
        
        phcmd = self.hist['commanded'][6, :]
        thcmd = self.hist['commanded'][7, :]
        pscmd = self.hist['commanded'][8, :]
        
        ax = fig.add_subplot(6,2,7)
        if not np.isnan(phcmd).any():
            ax.plot(tvec, np.degrees(phcmd), 'r-', label='command')
        ax.plot(tvec, np.degrees(ph), 'b-', label='truth')
        ax.set_ylabel(r'$\phi$'); ax.grid()

        ax = fig.add_subplot(6,2,9)
        if not np.isnan(thcmd).any():
            ax.plot(tvec, np.degrees(thcmd), 'r-', label='command')
        ax.plot(tvec, np.degrees(th), 'b-', label='truth')
        ax.set_ylabel(r'$\theta$'); ax.grid()
        
        ax = fig.add_subplot(6,2,11)
        if not np.isnan(pscmd).any():
            ax.plot(tvec, np.degrees(pscmd), 'r-', label='command')
        ax.plot(tvec, np.degrees(ps), 'b-', label='truth')
        ax.set_ylabel(r'$\psi$'); ax.grid()
        
        #
        # Angular Rates
        #
        
        # for convenience
        p = self.hist['truth'][9, :]
        q = self.hist['truth'][10, :]
        r = self.hist['truth'][11, :]

        pcmd = self.hist['commanded'][9, :]
        qcmd = self.hist['commanded'][10, :]
        rcmd = self.hist['commanded'][11, :]
        
        ax = fig.add_subplot(6,2,8)
        if not np.isnan(pcmd).any():
            ax.plot(tvec, pcmd, 'r-', label='command')
        ax.plot(tvec, p, 'b-', label='truth')
        ax.set_ylabel('p'); ax.grid()

        ax = fig.add_subplot(6,2,10)
        if not np.isnan(qcmd).any():
            ax.plot(tvec, qcmd, 'r-', label='command')
        ax.plot(tvec, q, 'b-', label='truth')
        ax.set_ylabel('q'); ax.grid()
        
        ax = fig.add_subplot(6,2,12)
        if not np.isnan(rcmd).any():
            ax.plot(tvec, rcmd, 'r-', label='command')
        ax.plot(tvec, r, 'b-', label='truth')
        ax.set_ylabel('r'); ax.grid()
        
        plt.show()
        
        #
        # Control Effort
        #
        
        thrust = self.hist['u'][0, :]
        tau_ph = self.hist['u'][1, :]
        tau_th = self.hist['u'][2, :]
        tau_ps = self.hist['u'][3, :]
        
        fig = plt.figure(figsize=(12,3))
        fig.subplots_adjust(wspace=0.25)
        fig.suptitle('Control Effort', fontsize=16)
        
        ax = fig.add_subplot(221)
        ax.plot(tvec, thrust, 'g-')
        ax.set_ylabel('Thrust'); ax.grid()
        
        ax = fig.add_subplot(222)
        ax.plot(tvec, tau_ph, 'g-')
        ax.set_ylabel(r'$\tau_\phi$'); ax.grid()
        
        ax = fig.add_subplot(223)
        ax.plot(tvec, tau_th, 'g-')
        ax.set_ylabel(r'$\tau_\theta$'); ax.grid()
        
        ax = fig.add_subplot(224)
        ax.plot(tvec, tau_ps, 'g-')
        ax.set_ylabel(r'$\tau_\psi$'); ax.grid()
        
        plt.show()

class Quadrotor(object):
    """Quadrotor
        
    This class models the physical quadrotor vehicle evolving in SE(3).
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
        Jxx = 0.060224; Jyy = 0.122198; Jzz = 0.132166
        self.I = np.array([[Jxx,0,0],
                           [0,Jyy,0],
                           [0,0,Jzz]])
        
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
        return self.mass*self.g*np.array([[      -np.sin(theta)     ],
                                          [np.cos(theta)*np.sin(phi)],
                                          [np.cos(theta)*np.cos(phi)]])
    
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
        # f = lambda v: (1/self.mass)*(self.Fg(0,0) - Rot_i_to_b(ph,th,ps).T.dot(T) - self.Mu.dot(v)) # inertial
        f = lambda v: (1/self.mass)*(self.Fg(0,0) - rotation(ph,th,ps).T.dot(T) - self.Mu.dot(v)) # inertial
        self.v = rk4(f, self.v, dt)
        # print(np.array_equal(Rot_i_to_b(ph,th,ps),rotation(ph,th,ps)))
        # print('\n')
        # print(Rot_i_to_b(ph,th,ps))
        # print('\n')
        # print(rotation(ph,th,ps))
        # Rotational
        f = lambda omega: np.linalg.inv(self.I).dot((-np.cross(omega, self.I.dot(omega), axis=0) + M))
        self.omega = rk4(f, self.omega, dt)
        
        # update control input
        u = np.hstack((T[2], M.flatten()))
        return u

class DirtyDerivative:
    """Dirty Derivative
    
    Provides a first-order derivative of a signal.
    
    This class creates a filtered derivative based on a
    band-limited low-pass filter with transfer function:
    
        G(s) = s/(tau*s + 1)
        
    This is done because a pure differentiator (D(s) = s)
    is not realizable.    
    """
    def __init__(self, order=1, tau=0.05):
        # time constant of dirty-derivative filter.
        # Higher leads to increased smoothing.
        self.tau = tau
        
        # Although this class only provides a first-order
        # derivative, we use this parameter to know how
        # many measurements to ignore so that the incoming
        # data is smooth and stable. Otherwise, the filter
        # would be hit with a step function, causing
        # downstream dirty derivatives to be hit with very
        # large step functions.
        self.order = order
        
        # internal memory for lagged signal value
        self.x_d1 = None
        
        # Current value of derivative
        self.dxdt = None
        
    def update(self, x, Ts):
        # Make sure to store the first `order` measurements,
        # but don't use them until we have seen enough
        # measurements to produce a stable output
        if self.order > 0:
            self.order -= 1
            self.x_d1 = x
            return np.zeros(x.shape)        
        
        # Calculate digital derivative constants
        a1 = (2*self.tau - Ts)/(2*self.tau + Ts)
        a2 = 2/(2*self.tau + Ts)
        
        if self.dxdt is None:
            self.dxdt = np.zeros(x.shape)
        
        # calculate derivative
        self.dxdt = a1*self.dxdt + a2*(x - self.x_d1)
        
        # store value for next time
        self.x_d1 = x
                
        return self.dxdt

class Sensor(object):
    """Sensor
    
    An abstract base class for all sensors.
    """
    def __init__(self):
        self.name = "Abstract Sensor"
        
    def __str__(self):
        return self.name
    
    def read(self, quad, n, Ts):
        return 0

class SensorManager(object):
    """Sensor Manager
    """
    def __init__(self):
        # create a list for sensor objects
        self.sensors = []
    
    def register(self, sensor):
        self.sensors += [sensor]
    
    def get_data_packet(self, quad, i, Ts):
        # dictionary of sensor data, keyed by sensor name
        pkt = {}
        
        for s in self.sensors:
            pkt[s.name] = s.read(quad, i, Ts)
            
        return pkt

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



if __name__ == "__main__":

    # Instantiate a quadrotor model with the given initial conditions
    quad = Quadrotor(r=np.array([[0],[0],[-10]]),
                     v=np.array([[0],[0],[0]]),
                   Phi=np.array([[-0.1],[0],[0]]))

    # Instantiate a tracking sliding mode controller
    ctrl = SMC()

    # Setup a setpoint commander
    cmdr = Commander(default=True)
    cmdr.attitude(np.array([0,0,np.pi/2]))

    def set_position(i, Ts):
        f = 0.25
        # x = np.sin(2*np.pi*f*i*Ts)
        x = np.cos(2*np.pi*f*i*Ts)
        return np.array([x, 2, -1])

    cmdr.position(set_position)

    # Run the simulation
    sim = Simulator(quad, ctrl, cmdr=cmdr)
    sim.run(20, Ts=0.01)
    sim.plot()