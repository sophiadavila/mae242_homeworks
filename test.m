
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

rho_vals = zeros(15,1);

for T = 1:15

    % Initial condition
    P = Q;

    % Riccati recursion
    for d = 2:T

        K = -(R + B'*P*B)^(-1) * B'*P*A;

        P = Q + K'*R*K + (A + B*K)'*P*(A + B*K);

    end

    % Gain corresponding to this horizon
    K_T = -(R + B'*P*B)^(-1) * B'*P*A;

    % Closed-loop matrix
    Acl = A + B*K_T;

    % Eigenvalues
    eigvals = eig(Acl);

    % Spectral radius
    rho_vals(T) = max(abs(eigvals));

    fprintf('T = %2d   Spectral Radius = %.4f', T, rho_vals(T));

    if rho_vals(T) < 1
        fprintf('   Stable\n');
    else
        fprintf('   Unstable\n');
    end

end

% Plot
figure;
plot(1:15, rho_vals, 'o-', 'LineWidth', 2);
hold on;
yline(1,'r--','LineWidth',2);

xlabel('Horizon T');
ylabel('Spectral Radius');
title('Spectral Radius of A + BK_T vs Horizon T');
grid on;