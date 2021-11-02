mutable struct Problem
    sigma::Function
    dsigma::Function
    eta::Function
    deta::Function
    DJW::Function
    DJmu::Function
    DJY::Function
    Vf::Function
    AdjVf::Function
    DVfK::Function
    DVfb::Function
    C
    function Problem(sigma,eta, C)
        dsigma = x -> ForwardDiff.derivative(sigma,x)
        deta = x -> ForwardDiff.derivative(eta,x)
        DJW(YN,W,mu,C) = YN' * (deta.(YN*W.+mu) .* (eta.(YN*W.+mu).-C))
        DJmu(YN,W,mu,C) = deta.(YN*W.+mu)' * (eta.(YN*W.+mu)-C)

        DJY(Y,W,mu,C) = (deta.(Y*W.+mu) .* (eta.(Y*W.+mu).-C))*W';

        Vf(Y,K,b) = sigma.(Y*K.+b');
        AdjVf(Y,P,K,b) = -(P .* dsigma.(Y*K.+b'))*K';

        DVfK(Y,P,K,b) = Y'*(dsigma.(Y*K.+b') .* P);
        DVfb(Y,P,K,b) = vec(sum(dsigma.(Y*K.+b') .* P, dims = 1));
        return new(sigma,dsigma,eta,deta,DJW,DJmu,DJY,Vf,AdjVf,DVfK,DVfb,C)
    end
end