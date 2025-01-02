function gradD=Lv(H, Hdstar, dist)
% LV computes the L operator of of a function of two variables with respect
% to the first argument. Specifically, it computes Lv of the EE-distance
% between H and Hd*, the nearest point in the curve. The output is a 
% "gradient" of the function that accounts for the inherent manifold 
% structure. It is a 1x6 vector.
I = eye(6);
gradD = zeros(1, 6);
eps=1e-3;
for i=1:6
    ei = I(:, i);
    variation = expse3(S(ei) * eps) * H;
    gradD(i) = (EEdist(variation, Hdstar) - dist) / eps;
end
end