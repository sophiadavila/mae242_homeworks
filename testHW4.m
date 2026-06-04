clear; clc; close all;

A = [ 1    0.4   0     0;
     -0.6  1     0.4   0;
      0    0.4   1    -0.6;
      0    0     0.4   1 ];

B = [1;
     0;
     0;
     0];

C = [0 0 0 1];

Q = C'*C;
R = 1;

P = Q;

spectral_radius = zeros(15,1);

for T = 1:15

    K = -(R + B'*P*B)^(-1) * B'*P*A;

    P = Q + K'*R*K + (A + B*K)'*P*(A + B*K);

    Acl = A + B*K;

    spectral_radius(T) = max(abs(eig(Acl)));

    fprintf('T = %2d   spectral radius = %.4f', T, spectral_radius(T));
    if spectral_radius(T) < 1
        fprintf(" Stable\n")
    else fprintf(" Unstable\n")
    end
end

figure;
plot(1:15, spectral_radius, '-o');
hold on;
yline(1,'r--'); % stability
xlabel('T');Z
ylabel('Spectral Radius');
grid on;

% clear; clc; close all;
% 
% A = [ 1    0.4   0     0;
%      -0.6  1     0.4   0;
%       0    0.4   1    -0.6;
%       0    0     0.4   1 ];
% 
% B = [1;
%      0;
%      0;
%      0];
% 
% C = [0 0 0 1];
%
% R = 1;
% 
% rhos = 1:-0.1:0.1;
% 
% Ts_vals = zeros(length(rhos),1);
% 
% 
% for r = 1:length(rhos)
% 
%     rho_q = rhos(r);
% 
%     Q = rho_q*(C'*C);
% 
%     P = Q;
% 
%     Ts = NaN;
% 
% 
%     for T = 1:15
% 
%         K = -(R + B'*P*B)^(-1) * B'*P*A;
% 
%         Acl = A + B*K;
% 
%         spectral_radius = max(abs(eig(Acl)));
% 
%         %fprintf('rho = %.1f   T = %2d   spectral radius = %.4f\n', rho_q, T, spectral_radius);
% 
%         if spectral_radius < 1 && isnan(Ts)
% 
%             Ts = T;
%             fprintf('rho = %.1f  Ts = %2d\n', rho_q,Ts);
% 
%         end
% 
%         P = Q + K'*R*K + (A + B*K)'*P*(A + B*K);
% 
%     end
% 
%     Ts_vals(r) = Ts;
% 
% end
% 
% 
% figure;
% 
% plot(rhos, Ts_vals,'o-');
% xlabel('\rho');
% ylabel('T_s(\rho)');