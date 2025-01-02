function xi = invS(liealg)
% INVS maps an element of se(3), a 4x4 matrix, into a 6-dimensional vector.
% The first 3 elements of xi are the linear velocity, and the last 3 are
% the angular velocity.
%   xi = S(A) maps a matrix A into a twist xi.
if ~isequal(size(liealg), size(zeros(4,4)))
    error('liealg must be a [4 4] matrix, not %s\n', mat2str(size(liealg)))
end
xi = zeros(6,1);
xi(1)=liealg(1,4);
xi(2)=liealg(2,4);
xi(3)=liealg(3,4);
xi(4)=-liealg(2,3);
xi(5)=liealg(1,3);
xi(6)=-liealg(1,2);
end