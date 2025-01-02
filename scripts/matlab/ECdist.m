function [dist, min_index]=ECdist(H, Hd)
% ECDIST computes the Element-Curve distance between the state H in SE(3)
% and a parametrized curve Hd in SE(3). Uses a brute-force approach. Hd
% must have a dimension 4x4xN, i.e., it is a list of homogeneous
% transformation matrices.
% For each element in Hd, it computes the EE-distance and finds the minimum
% value. It finds the nearest point in the curve, Hd(min_index) and the
% respective EEdist(H, Hd(min_index))
%
%   [d, i]=ECdist(H, Hdist) returns the minimum distance between H and the
%   curve, as well as the index of the nearest point in the curve Hd.
%
%   See also EEDIST
[n, m, N] = size(Hd);
if n~=4 || m~=4
    error('Curve should have 4x4 matrices as elements')
end
min_index=0;
dist=inf;
for i=1:N
    d = EEdist(H, Hd(:,:,i));
    if d < dist
        dist = d;
        min_index=i;
    end
end


