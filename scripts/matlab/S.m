function liealg = S(xi)
% S maps a 6-dimensional vector into an element of se(3), a 4x4 matrix.
%
%   A = S(xi) maps twist xi into matrix A.
if length(xi) ~= 6
    error('xi must be a 6-dimensional vector, not %d-dimensional\n', length(xi))
end
liealg=zeros(4,4);
% Skew-symmetric part of se(3) -- angular related:
liealg(1, 2) = -xi(6);
liealg(1, 3) = xi(5);
liealg(2, 3) = -xi(4);
liealg = liealg - liealg.';
% Position related part of se(3):
liealg(1, 4) = xi(1);
liealg(2, 4) = xi(2);
liealg(3, 4) = xi(3);
end