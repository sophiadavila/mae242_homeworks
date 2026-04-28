clc; clear; close all;

% P = [0.8 0.2 0;
%      0.4 0.4 0.2;
%      0.2 0.6 0.2];

P = [1 0 0;
     0.4 0.4 0.2;
     0 0.6 0.4];

bel1 = [1 0 0];
bel2 = [0 1 0]; 
bel3 = [0 0 1];

k = 1000;

for i = 1:k
    bel1 = bel1 * P;
    bel2 = bel2 * P;
    bel3 = bel3 * P;
end

bel1
bel2
bel3