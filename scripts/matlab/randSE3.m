function H=randSE3()
% RANDSE3 returns a random homogeneous transformation matrix;
%   H=randSE3();
v=rand(6,1);
H=expse3(S(v));
end