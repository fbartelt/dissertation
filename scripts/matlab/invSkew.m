function w=invSkew(A)
% INVSKEW maps a skew-symmetric matrix into a 3d vector. It maps an element
% of the Lie algebra so(3) into an angular velocity in R^3.
% invSkew(skew(a)) = a
w=zeros(3, 1);
w(1)=-A(2,3);
w(2)=A(1,3);
w(3)=-A(1,2);
end