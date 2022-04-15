
%% Define an array of the staring points for the trajectorise here
IP=[-20,20;
%     -10,6;
%     -10,4;
%     -9,4;
%     -8,4;
%     -8,6;
%     -6,3;
%     -5,6;
%     -4,4;
%     -4,2;
%     -3,3;
%     -2,2;
%     -2,1;
    ];
IP1=IP;
IP1(:,2)=IP1(:,2)+4;
IP2=IP1;
%% End of the array for the starting points of trajectories
%%
IP=[IP1;IP2];