function H = expse3(liealg)
% EXPSE3 returns the exponential map exp:se(3)->SE(3). It maps the Lie
% algebra into a Lie group element explicitly. More precise than using
% expm() function
%    H=expse3(A) maps the Lie algebra element into a homogeneous
%    transformation matrix.
if ~isequal(size(liealg), size(zeros(4,4)))
    error('liealg must be a [4 4] matrix, not %s\n', mat2str(size(liealg)))
end
skewmat=liealg(1:3, 1:3);
linvel=liealg(1:3, 4);

% Computes the rotation matrix using Rodrigues' formula, and exp map
% based on explicit power series expansion:
theta = sqrt(skewmat(1, 2)^2 + skewmat(1, 3)^2 + skewmat(2, 3)^2);
if theta < 1e-6
    % If theta approx. 0, return R=I, p=v
    R=eye(3);
    p=linvel;
else
    R = eye(3) + (sin(theta)/theta) * skewmat + ((1 - cos(theta))/(theta^2)) * skewmat * skewmat;
    U = eye(3) + ((1 - cos(theta)) / (theta^2)) * skewmat + ((theta - sin(theta))) / (theta^3) * skewmat * skewmat;
    p = U * linvel;
end
H=[R p; zeros(1, 3) 1];
end