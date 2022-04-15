t=0:0.1:48;
x=1.20*t;
y=1.20*t;
z=2.40*t-0.05*t.^2;
figure(1)
plot3(x,y,z,'LineWidth',2.5)
set(gca,'FontSize',14, 'FontWeight','bold') 
 xlabel('x_1(t)','FontSize',14,'FontWeight','bold','Color','b')
 ylabel('x_2(t)','FontSize',14,'FontWeight','bold','Color','b')
 zlabel('x_3(t)','FontSize',14,'FontWeight','bold','Color','b')
 axis([0 60 0 60 0 30])
grid
t=0:0.1:48;
x=1.20*t.*cos(0.020*t);
y=3.60*t.*sin(0.062*t);
z=2.40*t-0.05*t.^2;
figure(2)
plot3(x,y,z,'LineWidth',2.5)
set(gca,'FontSize',14, 'FontWeight','bold') 
 xlabel('x_1(t)','FontSize',14,'FontWeight','bold','Color','k')
 ylabel('x_2(t)','FontSize',14,'FontWeight','bold','Color','k')
 zlabel('x_3(t)','FontSize',14,'FontWeight','bold','Color','k')
 %axis([0 60 0 60 0 30])
grid