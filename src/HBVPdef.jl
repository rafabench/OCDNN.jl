mutable struct Problem
    C # Labels'
end

sigma(x) = tanh.(x)
dsigma(x) = 1 ./ (cosh(x).^2)

eta(x) = exp(x) ./ (exp.(x)+1)
deta(x) = exp(x) ./ (exp.(x)+1).^2

DJW(YN,W,mu,C) = YN' * deta(YN*W+mu) .* (eta(YN*W+mu)-C)
DJmu(YN,W,mu,C) = deta(YN*W+mu)' .* (eta(YN*W+mu)-C)

DJY(YN,W,mu,C) = (deta(Y*W+mu) .* (eta(Y*W+mu)-C))*W';

Vf(Y,K,b) = sigma(Y*K+b);
AdjVf(Y,P,K,b) = -(P .* dsigma(Y*K+b))*K';

DVfK(Y,P,K,b) = Y'*(Problem.dsigma(Y*K+b) .* P);
DVfb(Y,P,K,b) = sum(Problem.dsigma(Y*K+b) .* P);