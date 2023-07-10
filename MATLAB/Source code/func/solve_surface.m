%% Solve plane equation coefficients
function [a,b,c,d] = solve_surface(V1,V2,a1,b1,c1)
syms x1 x2 x3
if V1(2)==V2(2)&&V1(3)==V2(3)
    a=0;
    b=1;
    [c,d]=solve(V1(2)+x2*V1(3)+x3==0,...
       b1+c1*x2==0,x2,x3);
    c=double(c);
    d=double(d);
else
    a=1;
    [b,c,d]=solve(a*V1(1)+x1*V1(2)+x2*V1(3)+x3==0,...
        a*V2(1)+x1*V2(2)+x2*V2(3)+x3==0,...
        a*a1+b1*x1+c1*x2==0,x1,x2,x3);
    b=double(b);
    c=double(c);
    d=double(d);
end
end