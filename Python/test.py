from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

class slidingModeControl:
    def __init__(self, i, path: bool):
        """
        Initialize the class, this inits all variables
        :i is the length of simulation
        """
        self.i = i
        self.path = path

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

        self.dt = 0.1
        self.v = np.full((12,),0)
        self.lam = np.diagflat([0.0001,0.0001,0.0001])
        self.eta = 2

        self.data = {'xCMD': [],
                     'yCMD': [],
                     'zCMD': [],
                     'xACT': [],
                     'yACT': [],
                     'zACT': [],
                     }
    
    def dX1(self, v) -> float: 
        return v[6]

    def dX2(self, v) -> float: 
        return v[7]

    def dX3(self, v) -> float: 
        return v[8]

    def dX4(self, v) -> float: 
        return v[9]

    def dX5(self, v) -> float: 
        return v[10]

    def dX6(self, v) -> float: 
        return v[11]

    def dX7(self, v, runge: bool) -> float:
        x4 = v[3]
        x5 = v[4]
        x6 = v[5]
        x7 = v[6]
        u1 = v[12]
        u2 = v[13]
        u3 = v[14]
        u4 = v[15]
        if runge:
            k1 = (1/self.m)*(np.cos(x4))*np.sin(x5)*np.cos(x6)*(u1 + u2 + u3 + u4)
            # print("y= ", y)
            # k2 = f(y + dt/2*k1)
            k2 = (1/self.m)*(np.cos(x4 + self.dt/2*k1))*np.sin(x5 + self.dt/2*k1)*np.cos(x6 + self.dt/2*k1)*((u1 + self.dt/2*k1) + (u2 + self.dt/2*k1) + (u3 + self.dt/2*k1) + (u4 + self.dt/2*k1))
            # k3 = f(y + dt/2*k2)
            k3 = (1/self.m)*(np.cos(x4 + self.dt/2*k2))*np.sin(x5 + self.dt/2*k2)*np.cos(x6 + self.dt/2*k2)*((u1 + self.dt/2*k2) + (u2 + self.dt/2*k2) + (u3 + self.dt/2*k2) + (u4 + self.dt/2*k2))
            # k4 = f(y + dt  *k3)
            k4 = (1/self.m)*(np.cos(x4 + self.dt*k3))*np.sin(x5 + self.dt*k3)*np.cos(x6 + self.dt*k3)*((u1 + self.dt*k3) + (u2 + self.dt*k3) + (u3 + self.dt*k3) + (u4 + self.dt*k3))
            x7 = x7 + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            # (1/self.m)*(np.cos(x4))*np.sin(x5)*np.cos(x6)*(u1 + u2 + u3 + u4)
        else:
            x7 = (1/self.m)*(np.cos(x4))*np.sin(x5)*np.cos(x6)*(u1 + u2 + u3 + u4)
        # for x in v:
        #     n += 1
        #     print(n, x)
        return x7

    def dX8(self, v, runge: bool) -> float:
        x4 = v[3]
        x6 = v[5]
        x8 = v[7]
        u1 = v[12]
        u2 = v[13]
        u3 = v[14]
        u4 = v[15]
        if runge:
            k1 = -(1/self.m)*np.cos(x4)*np.sin(x6)*(u1 + u2 + u3 + u4)
            # k2 = f(y + dt/2*k1)
            k2 = -(1/self.m)*np.cos(x4 + self.dt/2*k1)*np.sin(x6 + self.dt/2*k1)*((u1 + self.dt/2*k1) + (u2 + self.dt/2*k1) + (u3 + self.dt/2*k1) + (u4 + self.dt/2*k1))
            # k3 = f(y + dt/2*k2)
            k3 = -(1/self.m)*np.cos(x4 + self.dt/2*k2)*np.sin(x6 + self.dt/2*k2)*((u1 + self.dt/2*k2) + (u2 + self.dt/2*k2) + (u3 + self.dt/2*k2) + (u4 + self.dt/2*k2))
            # k4 = f(y + dt  *k3)
            k4 = -(1/self.m)*np.cos(x4 + self.dt*k3)*np.sin(x6 + self.dt*k3)*((u1 + self.dt*k3) + (u2 + self.dt*k3) + (u3 + self.dt*k3) + (u4 + self.dt*k3))
            x8 = x8 + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            x8 = -(1/self.m)*np.cos(x4)*np.sin(x6)*(u1 + u2 + u3 + u4)
        return x8

    def dX9(self, v, runge: bool) -> float:
        x5 = v[4]
        x6 = v[5]
        x9 = v[8]
        u1 = v[12]
        u2 = v[13]
        u3 = v[14]
        u4 = v[15]
        if runge:
            k1 = (1/self.m)*np.cos(x5)*np.cos(x6)*(u1+ u2 + u3 + u4) - self.g
            k2 = (1/self.m)*np.cos(x5 + self.dt/2*k1)*np.cos(x6 + self.dt/2*k1)*((u1 + self.dt/2*k1) + (u2 + self.dt/2*k1) + (u3 + self.dt/2*k1) + (u4 + self.dt/2*k1)) - self.g
            k3 = (1/self.m)*np.cos(x5 + self.dt/2*k2)*np.cos(x6 + self.dt/2*k2)*((u1 + self.dt/2*k2) + (u2 + self.dt/2*k2) + (u3 + self.dt/2*k2) + (u4 + self.dt/2*k2)) - self.g
            k4 = (1/self.m)*np.cos(x5 + self.dt*k3)*np.cos(x6 + self.dt*k3)*((u1 + self.dt*k3) + (u2 + self.dt*k3) + (u3 + self.dt*k3) + (u4 + self.dt*k3)) - self.g
            x9 = x9 + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            x9 = (1/self.m)*np.cos(x5)*np.cos(x6)*(u1+ u2 + u3 + u4) - self.g
        return x9

    def dX10(self, v, runge: bool) -> float:
        x5 = v[4]
        x6 = v[5]
        x10 = v[9]
        x11 = v[10]
        x12 = v[11]
        u1 = v[12]
        u3 = v[14]
        if runge:
            k1 = (self.Iy/self.Iz)*x10*x11*np.cos(x5)*np.sin(x5) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*x11*x12*np.cos(x5) + \
                    ((2*self.Ix**2 + self.Iz**2 -3*self.Ix*self.Iz)/(self.Ix*self.Iz))*x10*x12*np.cos(x6)*np.sin(x6) + \
                    (self.a/self.Ix)*np.cos(x5)*np.sin(x6)*u1 -(self.a/self.Ix)*np.cos(x5)*np.sin(x6)*u3
            k2 = (self.Iy/self.Iz)*(x10 + self.dt/2*k1)*(x11 + self.dt/2*k1)*np.cos(x5 + self.dt/2*k1)*np.sin(x5 + self.dt/2*k1) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*(x11 + self.dt/2*k1)*(x12 + self.dt/2*k1)*np.cos(x5 + self.dt/2*k1) + \
                    ((2*self.Ix**2 + self.Iz**2 -3*self.Ix*self.Iz)/(self.Ix*self.Iz))*(x10 + self.dt/2*k1)*(x12 + self.dt/2*k1)*np.cos(x6 + self.dt/2*k1)*np.sin(x6 + self.dt/2*k1) + \
                    (self.a/self.Ix)*np.cos(x5 + self.dt/2*k1)*np.sin(x6 + self.dt/2*k1)*(u1 + self.dt/2*k1) -(self.a/self.Ix)*np.cos(x5 + self.dt/2*k1)*np.sin(x6 + self.dt/2*k1)*(u3 + self.dt/2*k1)
            k3 = (self.Iy/self.Iz)*(x10 + self.dt/2*k2)*(x11 + self.dt/2*k2)*np.cos(x5 + self.dt/2*k2)*np.sin(x5 + self.dt/2*k2) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*(x11 + self.dt/2*k2)*(x12 + self.dt/2*k2)*np.cos(x5 + self.dt/2*k2) + \
                    ((2*self.Ix**2 + self.Iz**2 -3*self.Ix*self.Iz)/(self.Ix*self.Iz))*(x10 + self.dt/2*k2)*(x12 + self.dt/2*k2)*np.cos(x6 + self.dt/2*k2)*np.sin(x6 + self.dt/2*k2) + \
                    (self.a/self.Ix)*np.cos(x5 + self.dt/2*k2)*np.sin(x6 + self.dt/2*k2)*(u1 + self.dt/2*k2) -(self.a/self.Ix)*np.cos(x5 + self.dt/2*k2)*np.sin(x6 + self.dt/2*k2)*(u3 + self.dt/2*k2)
            k4 = (self.Iy/self.Iz)*(x10 + self.dt*k3)*(x11 + self.dt*k3)*np.cos(x5 + self.dt*k3)*np.sin(x5 + self.dt*k3) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*(x11 + self.dt*k3)*(x12 + self.dt*k3)*np.cos(x5 + self.dt*k3) + \
                    ((2*self.Ix**2 + self.Iz**2 -3*self.Ix*self.Iz)/(self.Ix*self.Iz))*(x10 + self.dt*k3)*(x12 + self.dt*k3)*np.cos(x6 + self.dt*k3)*np.sin(x6 + self.dt*k3) + \
                    (self.a/self.Ix)*np.cos(x5 + self.dt*k3)*np.sin(x6 + self.dt*k3)*(u1 + self.dt*k3) -(self.a/self.Ix)*np.cos(x5 + self.dt*k3)*np.sin(x6 + self.dt*k3)*(u3 + self.dt*k3)
            x10 = x10 + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            x10 = (self.Iy/self.Iz)*x10*x11*np.cos(x5)*np.sin(x5) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*x11*x12*np.cos(x5) + \
                    ((2*self.Ix**2 + self.Iz**2 -3*self.Ix*self.Iz)/(self.Ix*self.Iz))*x10*x12*np.cos(x6)*np.sin(x6) + \
                    (self.a/self.Ix)*np.cos(x5)*np.sin(x6)*u1 -(self.a/self.Ix)*np.cos(x5)*np.sin(x6)*u3
        return x10

    def dX11(self, v, runge: bool) -> float:
        x5 = v[4]
        x6 = v[5]
        x10 = v[9]
        x11 = v[10]
        x12 = v[11]
        u1 = v[12]
        u3 = v[14]
        if runge:
            k1 = ((self.Iz - 2*self.Ix)/self.Ix)*x10*x11*np.cos(x5) + \
                    (((self.Iz-self.Ix)*(self.Ix-self.Iy+self.Iz))/(self.Ix*self.Iz))*x11*x12*np.cos(x6)*np.sin(x6) + \
                    -(self.a/self.Ix)*np.cos(x6)*u1 + (self.a/self.Ix)*np.cos(x6)*u3
            k2 = ((self.Iz - 2*self.Ix)/self.Ix)*(x10 + self.dt/2*k1)*(x11 + self.dt/2*k1)*np.cos(x5 + self.dt/2*k1) + \
                    (((self.Iz-self.Ix)*(self.Ix-self.Iy+self.Iz))/(self.Ix*self.Iz))*(x11 + self.dt/2*k1)*(x12 + self.dt/2*k1)*np.cos(x6 + self.dt/2*k1)*np.sin(x6 + self.dt/2*k1) + \
                    -(self.a/self.Ix)*np.cos(x6 + self.dt/2*k1)*(u1 + self.dt/2*k1) + (self.a/self.Ix)*np.cos(x6 + self.dt/2*k1)*(u3 + self.dt/2*k1)
            k3 = ((self.Iz - 2*self.Ix)/self.Ix)*(x10 + self.dt/2*k2)*(x11 + self.dt/2*k2)*np.cos(x5 + self.dt/2*k2) + \
                    (((self.Iz-self.Ix)*(self.Ix-self.Iy+self.Iz))/(self.Ix*self.Iz))*(x11 + self.dt/2*k2)*(x12 + self.dt/2*k2)*np.cos(x6 + self.dt/2*k2)*np.sin(x6 + self.dt/2*k2) + \
                    -(self.a/self.Ix)*np.cos(x6 + self.dt/2*k2)*(u1 + self.dt/2*k2) + (self.a/self.Ix)*np.cos(x6 + self.dt/2*k2)*(u3 + self.dt/2*k2)
            k4 = ((self.Iz - 2*self.Ix)/self.Ix)*(x10 + self.dt*k3)*(x11 + self.dt*k3)*np.cos(x5 + self.dt*k3) + \
                    (((self.Iz-self.Ix)*(self.Ix-self.Iy+self.Iz))/(self.Ix*self.Iz))*(x11 + self.dt*k3)*(x12 + self.dt*k3)*np.cos(x6 + self.dt*k3)*np.sin(x6 + self.dt*k3) + \
                    -(self.a/self.Ix)*np.cos(x6 + self.dt*k3)*(u1 + self.dt*k3) + (self.a/self.Ix)*np.cos(x6 + self.dt*k3)*(u3 + self.dt*k3)
            x11 = x11 + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            x11 = ((self.Iz - 2*self.Ix)/self.Ix)*x10*x11*np.cos(x5) + \
                    (((self.Iz-self.Ix)*(self.Ix-self.Iy+self.Iz))/(self.Ix*self.Iz))*x11*x12*np.cos(x6)*np.sin(x6) + \
                    -(self.a/self.Ix)*np.cos(x6)*u1 + (self.a/self.Ix)*np.cos(x6)*u3

        return x11

    def dX12(self, v, runge: bool) -> float:
        x5 = v[4]
        x10 = v[9]
        x11 = v[10]
        x12 = v[11]
        u2 = v[13]
        u4 = v[15]
        if runge:
            k1 = (self.Iy/self.Ix)*x10*x11*np.cos(x5) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*x11*x12*np.cos(x5)*np.sin(x5) + \
                    (self.a/self.Ix)*u2 + (-self.a/self.Ix) *u4
            k2 = (self.Iy/self.Ix)*(x10 + self.dt/2*k1)*(x11 + self.dt/2*k1)*np.cos(x5 + self.dt/2*k1) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*(x11 + self.dt/2*k1)*(x12 + self.dt/2*k1)*np.cos(x5 + self.dt/2*k1)*np.sin(x5 + self.dt/2*k1) + \
                    (self.a/self.Ix)*(u2 + self.dt/2*k1) + (-self.a/self.Ix) *(u4 + self.dt/2*k1)
            k3 = (self.Iy/self.Ix)*(x10 + self.dt/2*k2)*(x11 + self.dt/2*k2)*np.cos(x5 + self.dt/2*k2) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*(x11 + self.dt/2*k2)*(x12 + self.dt/2*k2)*np.cos(x5 + self.dt/2*k2)*np.sin(x5 + self.dt/2*k2) + \
                    (self.a/self.Ix)*(u2 + self.dt/2*k2) + (-self.a/self.Ix) *(u4 + self.dt/2*k2)
            k4 = (self.Iy/self.Ix)*(x10 + self.dt*k3)*(x11 + self.dt*k3)*np.cos(x5 + self.dt*k3) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*(x11 + self.dt*k3)*(x12 + self.dt*k3)*np.cos(x5 + self.dt*k3)*np.sin(x5 + self.dt*k3) + \
                    (self.a/self.Ix)*(u2 + self.dt*k3) + (-self.a/self.Ix) *(u4 + self.dt*k3)
            x12 = x12 + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            x12 = (self.Iy/self.Ix)*x10*x11*np.cos(x5) + \
                    ((self.Ix - self.Iy + self.Iz)/self.Iz)*x11*x12*np.cos(x5)*np.sin(x5) + \
                    (self.a/self.Ix)*u2 + (-self.a/self.Ix) *u4
        return x12

    def rk4(self, f: Any, y: Any, dt: float):
        """
        Runge-Kutta 4th Order
        Solves an differential equation of the form dy/dt = f(y)

        """
        k1 = f(y)
        # print("y= ", y)
        k2 = f(y + dt/2*k1)
        k3 = f(y + dt/2*k2)
        k4 = f(y + dt  *k3)
        return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    def P1(self, t: int, p: int = 0):
        """
        This is defined self.path #1 from ICEME paper
        """
        if p == 0:
            """
            0th derivative i.e. original function
            """
            p1 = 1.20*self.t
            p2 = 1.20*self.t
            p3 = 2.50*self.t-0.05*self.t**2

        elif p == 1:
            """
            First derivative
            """
            p1 = self.t
            p2 = self.t
            p3 = 2.50 - 0.1*self.t

        else:
            """
            Second derivative
            """
            p1 = 0
            p2 = 0
            p3 = 0
        return np.array([p1, p2, p3])

    def P2(self, t: int, p: int = 0):
        """
        This is defined self.path #2 from ICEME paper
        """
        if p == 0:
            """
            0th derivative i.e. original function
            """
            p1 = 1.20*self.t*np.cos(0.02*self.t)
            p2 = 3.60*self.t*np.sin(0.062*self.t)
            p3 = 2.40*self.t-0.05*self.t**2

        elif p == 1:
            """
            First derivative
            """
            p1 = 1.20*np.cos(0.02*self.t) - 0.024*self.t*np.sin(0.02*self.t)
            p2 = 3.60*np.sin(0.062*self.t) + 0.2232*self.t*np.cos(0.062*self.t)
            p3 = 2.40-0.1*self.t
        else:
            """
            Second derivative
            """
            p1 = -0.048*np.sin(0.02*self.t) - 0.000484*self.t*np.cos(0.02*self.t)
            p2 = 0.4464*np.cos(0.062*self.t) - 0.0138384*self.t*np.sin(0.062*self.t)
            p3 = -0.1
        return np.array([p1, p2, p3])

    def alpha1(self) -> float:
        return (1/self.m)*(np.cos(self.x4))*np.sin(self.x5)*np.cos(self.x6)

    def alpha2(self) -> float:
        return -(1/self.m)*np.cos(self.x4)*np.sin(self.x6)
    
    def alpha3(self) -> float:
        return (1/self.m)*np.cos(self.x5)*np.cos(self.x6)

    def sigma(self,idx: int, t: int) -> float:
        """
        :idx - int, index of sigma 1,2,3
        :t
        """
        if self.path:
            dp = self.P1(t,1)
            p = self.P1(t, 0)
        dp = self.P2(t, 1)
        p = self.P2(t, 0)
        # print(self.y[idx])
        # print(self.dy[idx].shape,self.y[idx].shape)
        sig = self.dy[idx] - dp[idx] + self.lam[idx,idx]*(self.y[idx] - p[idx])
        # print('sig=',sig, self.t)
        # print(sig)
        return sig

    def u(self):

        A = np.array([[self.alpha1(), self.alpha1(), self.alpha1(), self.alpha1()],
                    [self.alpha2(), self.alpha2(), self.alpha2(), self.alpha2()],
                    [self.alpha3(), self.alpha3(), self.alpha3(), self.alpha3()]])

        if self.path:
            ddp = self.P1(self.t, 2)
            dp = self.P1(self.t, 1)
        else:
            ddp = self.P2(self.t, 2)
            dp = self.P2(self.t, 1)
        
        self.y = np.array([self.x1, self.x2, self.x3])
        self.dy = np.array([self.x7, self.x8, self.x9])

        u_sum = []
        for x in range(0,3):
            sig = self.sigma(x,self.t)
            
            _A = A[x].reshape(1,4)
            
            if sig >= 0:
                # u_i = (-1/(np.dot(np.transpose(A[x]),A[x])))* np.transpose(A[x])*self.eta*sig
                # print(np.transpose(A).shape, np.transpose(A))
                # print(A.shape, A)
                # print(np.dot(np.transpose(A),A))
                u_i = -inv(np.dot(np.transpose(_A),_A))* np.transpose(_A)*self.eta*sig

                u_sum.append(u_i)
            elif sig == 0:
                u_i = 0
            else:
                
                u_i = (1/(np.dot(np.transpose(_A),_A)))* np.transpose(_A)*self.eta*sig
                u_sum.append(u_i)
                
        u_sum = np.sum(u_sum)
        
        u = np.dot(inv(np.dot(np.transpose(A),A)),np.dot(np.transpose(A),(ddp - np.dot(self.lam, (self.dy-dp)))) + u_sum)

        return u

    def update(self):
        """
        Each Iteration of t:
        1. get u
        2. solve dX in RK4
        3. store outputs in v
        4. add values to plot value holders
        """
        #place to store values for loop
        v = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        while self.t < self.i:
            print("iter=", self.t, np.round(v,3))
            u = self.u()

            if self.path:
                p = self.P1(self.t, 0)
            else:
                p = self.P2(self.t, 0)

            # dX1 - > x1 use old values for rk4
            f = lambda x: self.x7
            self.x1 = self.rk4(f, v[6], self.dt)
            # print("x1=" , self.x1)
            #dX2 - > x2
            f = lambda x: self.x8
            self.x2 = self.rk4(f, v[7], self.dt)
            #dX3 - > x3
            f = lambda x: self.x9
            self.x3 = self.rk4(f, v[8], self.dt)
            #dX4 -> x4
            f = lambda x: self.x10
            self.x4 = self.rk4(f, v[9], self.dt)
            #dX5 -> x5
            f = lambda x: self.x11
            self.x5 = self.rk4(f, v[10], self.dt)
            #dX6 -> x6
            f = lambda x: self.x12
            self.x6 = self.rk4(f, v[11], self.dt)

            self.x7 = self.dX7(v, True)
            self.x8 = self.dX8(v, True)
            self.x9 = self.dX9(v, True)
            self.x10 = self.dX10(v, True)
            self.x11 = self.dX11(v, True)
            self.x12 = self.dX12(v, True)
            
            #store new values for next loop
            v = [self.x1, 
                self.x2, 
                self.x3, 
                self.x4, 
                self.x5, 
                self.x6, 
                self.x7, 
                self.x8, 
                self.x9, 
                self.x10, 
                self.x11, 
                self.x12, 
                u[0], 
                u[1], 
                u[2], 
                u[3]]
            # print(v)
            self.data['xCMD'] += [self.x1]
            self.data['yCMD'] += [self.x2]
            self.data['zCMD'] += [self.x3]
            
            self.data['xACT'] += [p[0]]
            self.data['yACT'] += [p[1]]
            self.data['zACT'] += [p[2]]
            # print(self.p1,self.p2,self.p3)
            self.t += 1

    def plot(self):
        fialpha3D = plt.figure()
        ax3D = fialpha3D.add_subplot(projection='3d')

        ax3D.plot(self.data['xCMD'],self.data['yCMD'],self.data['zCMD'], label='actual')
        ax3D.plot(self.data['xACT'],self.data['yACT'],self.data['zACT'], label='desired')
        
        ax3D.axes.set_xlim3d(left = 60, right = 0)
        ax3D.axes.set_ylim3d(bottom = 0, top = 60)
        ax3D.axes.set_zlim3d(bottom = 0, top = 60)

        ax3D.legend()
        ax3D.axes.set_xlabel('X')
        ax3D.axes.set_ylabel('Y')
        ax3D.axes.set_zlabel('Z')
        plt.show()


if __name__ == "__main__":
    cntrl = slidingModeControl(50, True)
    cntrl.update()
    cntrl.plot()




# self.dX7(v)
#dX7 calculate these first
# f = lambda x: (1/self.m)*(np.cos(x[3]))*np.sin(x[4])*np.cos(x[5])*(x[11] + x[12] + x[13] + x[14])
# f = lambda x: (1/self.m)*(np.cos(x[0]))*np.sin(x[1])*np.cos(x[2])*(x[3] + x[4] + x[5] + x[6])
# self.x7 = self.rk4(f, [v[3],v[4],v[5],v[6],v[12],v[13],v[14],v[15]], self.dt)
# print(self.x7)
#dX8 
# f = lambda x4,x6,u1,u2,u3,u4: -(1/self.m)*np.cos(x4)*np.sin(x6)*(u1 + u2 + u3 + u4)
# self.x8 = self.rk4(self.dX8, v, self.dt)
#dX9
# f = lambda x5,x6: (1/self.m)*np.cos(x5)*np.cos(x6)
# self.x9 = self.rk4(self.dX9, v, self.dt)
#dX10
# self.x10 = self.rk4(self.dX10, v, self.dt)
#dX11
# self.x11 = self.rk4(self.dX11, v, self.dt)
#dX12
# self.x12 = self.rk4(self.dX12, v, self.dt)
