function [xi_N, xi_T, dist, min_index]=vectorfield(H, Hd, Hd_derivative)
% VECTORFIELD returns the normal and tangent components of the vector field
% strategy, where xi_N is the normal component that ensures convergence,
% and xi_T is the tangent component that ensures circulation.
% Note that both components must be multiplied by gains, that will need to
% be set outside this function for convenience.
%
%   [xi_N, xi_T, dist]=vectorfield(H, Hd) returns the components and 
%   minimum distanceusing an approximation for the tangent component.
%   [xi_N, xi_T, dist]=vectorfield(H, Hd, Hd_derivative) returns the 
%   components and minimum distance using an the precomputed curve 
%   derivative for an explicit tangent component computation.
[dist, min_index] = ECdist(H, Hd);
[~, ~, N] = size(Hd);
Hdstar = Hd(:, :, min_index);

% Compute normal component (ensures convergence)
xi_N = -Lv(H, Hdstar, dist).';

% Compute tangent component (ensures circulation). If the curve derivative
% was passed, use it for computation
if nargin == 3
    % Use curve derivative
    Hdd = Hd_derivative(:, :, min_index);
else
    % Use approximation
    eps=1e-3;
    if min_index == N
        Hdd = (Hd(:,:,1) - Hdstar) / eps;
    else
        Hdd = (Hd(:,:,min_index+1) - Hdstar) / eps;
    end
end
xi_T = invS(Hdd * inv(Hdstar));

end