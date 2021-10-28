mutable struct Gradient
    K::Array{Array{Float64,2},1} # Cell array of matrices containing control parameters
    b::Array{Array{Float64,1},1} # Cell array of row vectors containing control biases
    W::Array{Float64,1} # Array (channels,1) containing the control parameters of the final projection
    mu::Float64 # scalar bias for final projection (classifier)
    function Gradient(channels, nlayers)
        K = [zeros(channels,channels) for i in 1:nlayers]
        b = [zeros(channels) for i in 1:nlayers]
        W = zeros(channels)
        mu = 0.0
        return new(K,b,W,mu)
    end
end

function GradientCalc(Ctrls, rkmethod, C)

    S = RKstepper(Ctrls, rkmethod, HBVP);
    channels = Ctrls.channels;
    nlayers = Ctrls.nlayers;
    h = S.stepsize;
    s = rkmethod.s;

    last = nlayers+1;
    G = Gradient(channels, nlayers)
    G.W = DJW(S.Y[last],Ctrls.W,Ctrls.mu,C);
    G.mu = DJmu(S.Y[last],Ctrls.W,Ctrls.mu,C);
    E = randn(s)

    for k = 1:last-1
        for i = 1:s
            G.K[k] = G.K[k]+h*M.w[i]*DVfK(S.Ys[k,i],S.P[k+1],Ctrls.K[k],Ctrls.b[k]);
            G.b[k] = G.b[k]+h*M.w[i]*DVfb(S.Ys[k,i],S.P[k+1],Ctrls.K[k],Ctrls.b[k]);
            E[i]= -AdjVf(S.Ys[k,i],S.P[k+1],Ctrls.K[k],Ctrls.b[k]);
        end
        
        for m = 1:s-1
           for i = 1:s-m
               P = zeros(S.rows,channels);
               for j = i+1:s-m+1
                   P = P+h*M.At[i,j]*E[j];
               end
               G.K[k] = G.K[k] + h*M.w[i]*DVfK(S.Ys[k,i],P,Ctrls.K[k],Ctrls.b[k]);
               G.b[k] = G.b[k] + h*M.w[i]*DVfb(S.Ys[k,i],P,Ctrls.K[k],Ctrls.b[k]);
               E[i] =  -AdjVf(S.Ys[k,i],P,Ctrls.K[k],Ctrls.b[k]);
           end
        end
    end

    return G
end