function [a,b,c] = solve_line(V1,V2)
% Solving linear equations ax+by+c=0
% y=(y1-y2)/(x1-x2)*(x-x1)+y1;
x1=V1(1);y1=V1(2);
x2=V2(1);y2=V2(2);
if x1~=x2
    a=(y2-y1)/(x1-x2);
    b=1;
    c=(y1-y2)/(x1-x2)*x1-y1;
else
    a=1;
    b=0;
    c=-x1;
end
end