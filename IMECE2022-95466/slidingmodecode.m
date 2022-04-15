clear
clc
% In this example, the system is 
%
% dx1/dt=x2
% dx2/dt=-x1+0.4x2+u
%
%The control is 
%    /
%   |  4x1 if S>0
% u=|
%   | -4x` if S<0
%    \
%
%% Case S>0

u=0;
F=@(t,X) [X(2);-X(1)+0.4*X(2)+4*X(1)]; % define your system f(x)+g(x)u here
%F=@(t,X) [X(1)+0.4*X(2)*sin(X(1));0.2*X(2)^2+X(1)+0.2*X(1)^2]; % define your system f(x) here
%F=@(t,X) [0.5*X(1)*X(2)-3*X(1);-0.5*X(1)*X(2)+2*X(2)]; % define your system f(x) here
figure(1)
vectorfield;
pbaspect([1 1 1])
xlabel('x_1')
ylabel('x_2')
grid
%axis([-20,20,-20,20])
axis([-50,50,-50,50])
hold on
%% Define an array of the staring points for the trajectorise here
initialpoints;
plot(IP(:,1),IP(:,2),'kx','linewidth',1.5); % Mark the Starting point starting point
%IP=[15,10;25,20;35,30];
for k = 1:length(IP)
    [ts,ys] = ode45(F,[0,10],IP(k,:));
    plot(ys(:,1),ys(:,2),'r:','linewidth',1.5);
end
%% Case S<0
F=@(t,X) [X(2);-X(1)+0.4*X(2)-4*X(1)];  
%F=@(t,X) [X(1)+0.4*X(2)*sin(X(1));0.2*X(2)^2+X(1)-0.2*X(1)^2]; % define your system f(x) here
%F=@(t,X) [0.5*X(1)*X(2)-3*X(1);-0.5*X(1)*X(2)+2*X(2)]; % define your system f(x) here
for k = 1:length(IP)
    [ts,ys] = ode45(F,[0,10],IP(k,:));
    plot(ys(:,1),ys(:,2),'g-.','linewidth',1.5);
end
 hold off
 figure(2)
 pbaspect([1 1 1])
xlabel('x_1')
ylabel('x_2')
grid
axis([-20,20,-20,20])
%axis([-50,50,-10,10])
 hold on
%% Now Sliding mode control to bring all trajectories to S
u=0;
F1=@(x1,x2,t)  x2;
F2=@(x1,x2,u,t) -x1+0.4*x2+u;

%F1=@(x1,x2,t)  x1+0.4*x2*sin(x1);
%F2=@(x1,x2,u,t) 0.2*x2^2+x1+u;
DT=0.0001;
t=0:DT:150;
n=length(t);
rungekutta;  %% The switches control is implemented in this file
%legend('u(x_1) = 4x_1','u(x_1) =  -4x_1','Switched Control')


hold off
