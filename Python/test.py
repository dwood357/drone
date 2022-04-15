from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt

class slidingModeControl:
    def __init__(self, i):
        """
        Initialize the class, this inits all variables
        :i is the length of simulation
        """
        self.i = i
        self.x1 = 0
        self.x2 = 0
        self.x3 = 0
        self.x4 = 0.1
        self.x5 = 0.1
        self.x6 = 0.1
        self.x7 = 0.1
        self.x8 = 0.1
        self.x9 = 0.1
        self.x10 = 0.1
        self.x11 = 0.1
        self.x12 = 0.1
        # self.t = np.arange(0,i,1)
        self.t = 0

        self.a = 0.15
        self.g = 9.81
        self.m = 3.81
        self.Ix = 0.060224
        self.Iy = 0.122198
        self.Iz = 0.132166

        # self.lam = np.diagflat([1,1,1])
        self.eta = 1

        self.data = {'xCMD': [],
                     'yCMD': [],
                     'zCMD': [],
                     'xACT': [],
                     'yACT': [],
                     'zACT': [],
                     }

    def F(self):
        f = np.array([self.x7,
                    self.x8,
                    self.x9,
                    self.x10,
                    self.x11,
                    self.x12,
                    0,
                    0,
                    0,
                    (self.Iy/self.Iz)*self.x10*self.x11*np.cos(self.x5)*np.sin(self.x5) + 
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*self.x11*self.x12*np.cos(self.x5) +
                    ((2*self.Ix**2 + self.Iz**2 -3*self.Ix*self.Iz)/(self.Ix*self.Iz))*self.x10*self.x12*np.cos(self.x6)*np.sin(self.x6),
                    ((self.Iz - 2*self.Ix)/self.Ix)*self.x10*self.x11*np.cos(self.x5) +
                    (((self.Iz-self.Ix)*(self.Ix-self.Iy+self.Iz))/(self.Ix*self.Iz)) * self.x11*self.x12*np.cos(self.x6)*np.sin(self.x6),
                    (self.Iy/self.Ix)*self.x10*self.x11*np.cos(self.x5) +
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*self.x11*self.x12*np.cos(self.x5)*np.sin(self.x5)])
        # print(f.shape)
        return f

    def alpha1(self) -> float:
        return (1/self.m)*(np.cos(self.x4))*np.sin(self.x5)*np.cos(self.x6)

    def alpha2(self) -> float:
        return -(1/self.m)*np.cos(self.x4)*np.sin(self.x6)
    
    def alpha3(self) -> float:
        return (1/self.m)*np.cos(self.x5)*np.cos(self.x6)

    def G(self):
        g = np.array([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [self.alpha1(), self.alpha1(), self.alpha1(), self.alpha1()],
                    [self.alpha2(), self.alpha2(), self.alpha2(), self.alpha2()],
                    [self.alpha3(), self.alpha3(), self.alpha3(), self.alpha3()],
                    [(self.a/self.Ix)*np.cos(self.x5)*np.sin(self.x6), 0, -(self.a/self.Ix)*np.cos(self.x5)*np.sin(self.x6), 0],
                    [-(self.a/self.Ix)*np.cos(self.x6), 0, (self.a/self.Ix)*np.cos(self.x6), 0],
                    [0, self.a/self.Ix, 0, -self.a/self.Ix]])
        # print(g.shape)
        # print(g[6:9])
        return g
    
    def DX(self, u):
        """
        This is the state space model dx(X) = f(X) + g(X)*U + W
        F: (12,1)
        G: (12,4)
        u: (4,1)
        W: (12,1)
        return (12,1)
        :idx g(column)*u
        :u hand it calculated u
        """
        G = self.G()
        
        sumG = np.column_stack((G[:,0]*u[0],G[:,1]*u[1],G[:,2]*u[2],G[:,3]*u[3]))
        sumG = np.sum(sumG, axis=1)
        # print(sumG.shape)
        W = np.full((12,),0)
        W[8] = - self.g
        
        DX = self.F() + sumG + W
        # print(DX.shape)
        print(DX)
        return DX

    def rk4(self, f: Any, y: float, dt: float):
        """
        Runge-Kutta 4th Order
        Solves an differential equation of the form dy/dt = f(y)

        """
        k1 = f(y)
        k2 = f(y + dt/2*k1)
        k3 = f(y + dt/2*k2)
        k4 = f(y + dt  *k3)
        return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    def P1(self, t: int, p: int = 0):
        """
        This is defined path #1 from ICEME paper
        """
        if p == 0:
            """
            0th derivative i.e. original function
            """
            p1 = 1.20*self.t
            p2 = 1.20*self.t
            p3 = 2.50*self.t-0.05*self.t**2

        if p == 1:
            """
            First derivative
            """
            p1 = self.t
            p2 = self.t
            p3 = 2.50 - 0.1*self.t

        if p == 2:
            """
            Second derivative
            """
            p1 = 0
            p2 = 0
            p3 = 0
        self.x1 = p1
        self.x2 = p2
        self.x3 = p3
        return (p1, p2, p3)

    def P2(self, t: int, p: int = 0):
        """
        This is defined path #2 from ICEME paper
        """
        if p == 0:
            """
            0th derivative i.e. original function
            """
            p1 = 1.20*self.t*np.cos(0.02*self.t)
            p2 = 3.60*self.t*np.sin(0.062*self.t)
            p3 = 2.40*self.t-0.05*self.t**2

        if p == 1:
            """
            First derivative
            """
            p1 = 1.20*np.cos(0.02*self.t) - 0.024*self.t*np.sin(0.02*self.t)
            p2 = 3.60*np.sin(0.062*self.t) + 0.2232*self.t*np.cos(0.062*self.t)
            p3 = 2.40-0.1*self.t
        if p == 2:
            """
            Second derivative
            """
            p1 = -0.048*np.sin(0.02*self.t) - 0.000484*self.t*np.cos(0.02*self.t)
            p2 = 0.4464*np.cos(0.062*self.t) - 0.0138384*self.t*np.sin(0.062*self.t)
            p3 = -0.1
        self.x1 = p1
        self.x2 = p2
        self.x3 = p3
        return (p1, p2, p3)

    def sigma(self,idx: int, t: int, path: bool, y: List, dy: List) -> float:
        """
        :idx - int, index of sigma 1,2,3
        :t
        """
        if path:
            dp = self.P1(t,1)
            p = self.P1(t, 0)
        dp = self.P2(t, 1)
        p = self.P2(t, 0)
        self.lam = [1,1,1]
        sig = dy[idx] - dp[idx] + self.lam[idx]*(y[idx] - p[idx])
        # print(sig)
        return sig

    def u(self, path: bool, sig: int):

        A = np.array([[self.alpha1(), self.alpha1(), self.alpha1(), self.alpha1()],
                    [self.alpha2(), self.alpha2(), self.alpha2(), self.alpha2()],
                    [self.alpha3(), self.alpha3(), self.alpha3(), self.alpha3()]])
        # print(A[0], '\n')
        # print(A[1], '\n')
        # print(A[2])
        if path:
            ddp = self.P1(self.t, 2)
            dp = self.P1(self.t, 1)
        else:
            ddp = self.P2(self.t, 2)
            dp = self.P2(self.t, 1)
        y = [self.x1, self.x2, self.x3]
        dy = [self.x7, self.x8, self.x9]
        sig = self.sigma(0,self.t, False, y, dy)
        if sig >= 0:
            u = 1/(np.dot(A[0],A[0])) * np.dot(A[0],(ddp[0] - self.lam[0]*(dy[0]-dp[0]) - self.eta*sig))
            # u = np.linalg.inv(A.T * A) * A.T * (ddp - self.lam*(dy-dp) - self.eta*sig)
        elif sig == 0:
            u = 1/(np.dot(A[0],A[0])) * np.dot(A[0],(ddp[0] - self.lam[0]*(dy[0]-dp[0])))
            # u = np.linalg.inv(A.T * A) * A.T * (ddp - self.lam*(dy-dp))
        else:
            u = 1/(np.dot(A[0],A[0])) * np.dot(A[0],(ddp[0] - self.lam[0]*(dy[0]-dp[0]) + self.eta*sig))
            # u = np.linalg.inv(A.T * A) * A.T * (ddp - self.lam*(dy-dp) + self.eta*sig)
        # print(u.shape)
        return u
    def update(self):
        """
        Each Iteration of t:
        1. get sigma 0,1,2
        2. get u
        3. get dX
        4. solve dX in RK4
        5. store output
        """
        while self.t < self.i:
            u = self.u(False,0)
            self.DX(u)

            # self.rk4(self.DX, self.X, 0.01)

            self.data['xCMD'] += [self.x1]
            self.data['yCMD'] += [self.x2]
            self.data['zCMD'] += [self.x3]
            
            self.data['xACT'] += [self.x1]
            self.data['yACT'] += [self.x2]
            self.data['zACT'] += [self.x3]

            self.t += 1
        # print(self.data['Actual'])

    def plot(self):
        fialpha3D = plt.figure()
        ax3D = fialpha3D.add_subplot(projection='3d')
        
        # self.t = np.arange(0,600,1)
        # x,y,z = self.P1(self.t)
        # x = self.data['Actual'][]
        ax3D.plot(self.data['xCMD'],self.data['yCMD'],self.data['zCMD'], label='actual')
        
        # x,y,z = self.P2(self.t)
        # ax3D.plot(x,y,z, label='actual')

        # ax3D.scatter(xcmd,ycmd,zcmd, color='red', marker='o', label='CMD')
        ax3D.legend()
        plt.show()


if __name__ == "__main__":
    cntrl = slidingModeControl(100)
    cntrl.G()
    cntrl.update()
    cntrl.plot()