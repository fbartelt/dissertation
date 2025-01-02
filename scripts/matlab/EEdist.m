function dist=EEdist(V, W)
% EEDIST computes the Element-Element distance in SE(3). It computes the
% explicit expression for ||log(V^-1 W)||_F;
%   dist=EEdist(V, W) returns the adimensional distance between V and W
%
%   See also EEDISTVARS

Z = inv(V) * W;
[~, u, M, theta, ~, ~, ~]=EEdistvars(Z);
% Compute explicit form of ||log(V^-1 W)||_F
dist = sqrt(2*(theta^2) + u.'*M*u);
end