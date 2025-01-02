clear;close all;
%% Curve construction
% The curve is a circle on the ZX plane, rotating around the world yaxis
N = 500;
radius = 25e-2; % circle radius in meters
h0 = 0.5; % vertical displacement
Hd = zeros(4, 4, N);
Hd_derivative = zeros(4, 4, N);
% Curve parametrized by s=[0, 1];
s_samples = linspace(0, 1, N);
for i=1:N
    s = s_samples(i);
    theta = 2*pi*s;
    pd = [radius*cos(theta); 0; radius*sin(theta) + h0];
    Rd = rotxyz('y', theta);
    % Rd = eye(3);
    Hd(:,:,i) = [Rd pd; zeros(1, 3) 1];
    pdd = [-radius*sin(theta); 0; radius*cos(theta)];
    Rdd = skew([0; 1; 0]) * Rd;
    % Rdd = zeros(3);
    Hd_derivative(:,:,i)=[Rdd pdd; zeros(1, 3) 0];
end
% disp('Built curve')

%% Initial conditions
p0 = [0; 0; 0];
R0 = rotxyz('x', deg2rad(30)) * rotxyz('z', deg2rad(10));
H0 = [R0 p0; zeros(1, 3) 1];
H = H0;

%% Simulation
T = 1; %seconds
dt = 1e-3; %timestep
imax = round(T/dt);

phist = zeros(3, 1, imax);
disthist = zeros(imax, 1);
errposhist = zeros(imax, 1);
errorihist = zeros(imax, 1);

for i=1:imax
    progressbar(i, imax);
    [xi_N, xi_T, min_dist, min_index] = vectorfield(H, Hd, Hd_derivative);
    twist = kn(min_dist, 1, 5) * xi_N + kt(min_dist, 1, 0.8, 1) * xi_T;
    H = expse3(S(twist)*dt) * H;
    % NOTE o twist = [v omega], mas v nao eh a velocidade linear de fato do
    % objecto. Ela eh uma velocidade de um ponto virtual. Para computar as
    % verdadeiras velocidades angular e linear do objeto:
    angular_velocity = twist(4:end); % exatamente omega
    linear_velocity = cross(angular_velocity, H(1:3, 4)) + twist(1:3); % isso eh (omega x p) + v
    % Essas sao as verdadeiras velocidades linear e angular do efetuador
    % final.
    
    % Store variables
    p = H(1:3, 4);
    R = H(1:3, 1:3);
    phist(:, :, i) = p;
    disthist(i) = min_dist;
    pd = Hd(1:3, 4, min_index);
    Rd = Hd(1:3, 1:3, min_index);
    errposhist(i) = norm(p - pd);
    Rerr = Rd * inv(R);
    theta_ = atan2(1/(2*sqrt(2)) * norm(Rerr - inv(Rerr), "fro"), (trace(Rerr)-1)/2);
    errorihist(i) = theta_;
end
fprintf("Last Distance D: %.3f\n", min_dist);
fprintf("Last position error: %.3f cm\n", errposhist(end)*100);
fprintf("Last orientation error: %.3f deg\n", rad2deg(errorihist(end)));

%% Plotting
% Extract x, y, z components
x = squeeze(phist(1, 1, :)); % x-coordinates
y = squeeze(phist(2, 1, :)); % y-coordinates
z = squeeze(phist(3, 1, :)); % z-coordinates
x_Hd = squeeze(Hd(1, 4, :)); % x-coordinates
y_Hd = squeeze(Hd(2, 4, :)); % y-coordinates
z_Hd = squeeze(Hd(3, 4, :)); % z-coordinates

% 3D Plot
figure;
plot3(x_Hd, y_Hd, z_Hd, 'r--', 'LineWidth', 2)
hold on;
plot3(x, y, z, 'b-', 'LineWidth', 2); % Plot the trajectory
plot3(x(1), y(1), z(1), 'g^', 'MarkerSize', 12, 'MarkerFaceColor','green', 'DisplayName', 'Start'); % Mark the start
plot3(x(end), y(end), z(end), 'ro', 'MarkerSize', 12, 'MarkerFaceColor','red' ,'DisplayName', 'End'); % Mark the end
grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Trajectory');
legend('Curve', 'Trajectory', 'Start', 'End');

figure(2)
subplot(3, 1, 1)
time_vec = (1:imax)*dt;
plot(time_vec, disthist);
title('Distance along time')
ylabel('Distance D')
subplot(3, 1, 2)
plot(time_vec, errposhist*100);
ylabel('Pos. error (cm)')
subplot(3, 1, 3)
plot(time_vec, rad2deg(errorihist));
ylabel('Ori. error (deg)')
xlabel('Time (s)')

%% Gain Functions
function gain=kn(dist, k1, k2)
    gain = k1*tanh(k2*dist);
end

function gain=kt(dist, k1, k2, k3)
    gain = k1*(1 - k2*tanh(k3*dist));
end

function progressbar(current, imax)
    percent = floor(100 * current / imax);
    barWidth = 50;  % Width in characters

    pos = round(barWidth * current / imax);

    progressBar = ['[' repmat('=', 1, pos) repmat(' ', 1, barWidth - pos) ']'];
    if mod(current, ceil(imax / (barWidth * 100))) == 0 || current == imax
        clc;
        fprintf('\r%s %3d%%', progressBar, percent); % Use \r to overwrite the line
        if current == imax
            fprintf('\n'); % Move to the next line after the final update
        end
    end
end