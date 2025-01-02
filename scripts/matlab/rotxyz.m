function R=rotxyz(axis_, angle)
% ROTXYZ returns a rotation matrix with respect to axis_ 'x' 'y' or 'z' of
% angle radians.
if axis_=='x'
    R = [1 0 0; 0 cos(angle) -sin(angle); 0 sin(angle) cos(angle)];
elseif axis_ == 'y'
    R = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)];
else
    R = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1];
end
end