function [Q, u, M, theta, costheta, sintheta, alpha]=EEdistvars(Z)
% EEDISTVARS compute the elements necessary to compute the EE-distance and
% the L operator of the EE-distance (used in normal component). This is
% just an internal function to reduce computational effort.
% Q is the rotation matrix, u is the position vector, M is a matrix
% resulting from Cayley-Hamilton, theta is the angle of rotation, costheta
% is the computed cos(theta), sintheta is the computed sin(theta), and
% alpha is a variable used for Cayley-Hamilton computation.
%
% See also EEDIST
Q = Z(1:3, 1:3);
u = Z(1:3, 4);
costheta = 0.5 * (trace(Q) - 1);
sintheta = 1/(2*sqrt(2)) * norm(Q - inv(Q),"fro");
theta = atan2(sintheta, costheta);
% Recompute sin and cos for more precision
costheta = cos(theta);
sintheta = sin(theta);
% Compute Cayley-Hamilton matrix:
if costheta > 0.999
    % If theta close to 0, use the limit
    alpha = -1/12;
else
    alpha = (2 - 2*costheta - (theta^2))/(4 * (1 - costheta)^2);
end
M = alpha * (Q + inv(Q)) + (1 - 2*alpha) * eye(3);
end