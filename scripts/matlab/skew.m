function A=skew(w)
% SKEW maps a 3-dimensional vector into a skew-symmetric matrix. It maps an
% angular velocity in R^3 to the Lie algebra so(3);
if length(w) ~= 3
    error('The vector must be 3-dimensional, not %d-dimensional', length(w))
end
A = zeros(3);
A(1, 2) = -w(3);
A(1, 3) = w(2);
A(2, 3) = -w(1);
A = A - A.';
end