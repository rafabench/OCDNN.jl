function RKforwardstepper(Ctrls,rkmethod)

    nlayers = Ctrls.nlayers;
    channels = Ctrls.channels;
    h = Ctrls.stepsize;
    
    S = SolutionDef(h, rkmethod.s, Ctrls.Y0, channels, nlayers);
    last = nlayers+1;
    # Forward stepping
    F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
    for n = 1:nlayers
        for i = 1:rkmethod.s
           S.Ys[n,i]=S.Y[n];
           for j = 1:i-1
               S.Ys[n,i]=S.Ys[n,i].+h*rkmethod.A[i,j]*F[j];
           end
           F[i] = Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
        end
        S.Y[n+1] = S.Y[n];
        for i = 1:rkmethod.s
            S.Y[n+1]=S.Y[n+1]+h*rkmethod.w[i]*F[i];
        end
    end
    
    S.Classifier = eta(S.Y[last]*Ctrls.W .+ Ctrls.mu);
    return S
end