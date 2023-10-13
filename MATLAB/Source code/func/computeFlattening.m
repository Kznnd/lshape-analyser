function x=computeFlattening(A,b,L)
%given a Laplacian L, constraints A,b s.t. require Ax=b, find the solution
%to the equation given these boundary conditions, by solving the corresponding 
%KKT system

% L=V2A'*V2A;

% minimize ||Cx+d|| s.t. Ax=b
% minimize x^T L x s.t. Ax=b

n_vars = size(A,2);
n_eq = size(A,1);
M=[L A'; A sparse(n_eq,n_eq)];
rhs=[zeros(n_vars,1); b];
x_lambda = M \ rhs;
if ~isempty(find(isnan(x_lambda), 1)) 
    fprintf('The matrix is singular and cannot be solved\n');
    setGlobalx(2)
    x=-1;
    return
end
e=max(abs(M*x_lambda-rhs)); 
if e>1e-6
    fprintf('Linear system not solved');
    setGlobalx(2)
    x=-1;
    return
end
x = x_lambda(1:n_vars);
end