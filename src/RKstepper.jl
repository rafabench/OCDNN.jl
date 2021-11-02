function RKstepper(Ctrls, rkmethod, Problem)

    last = Ctrls.nlayers+1;
    S=RKforwardstepper(Ctrls,rkmethod, Problem); #Step forward
    S.P[last] = Problem.DJY(S.Y[last],Ctrls.W,Ctrls.mu,Problem.C); #Set right boundary value for P
    S=RKbackwardstepper(Ctrls, rkmethod, S, Problem); # Step backwards to obtain the Ps
    return S

end
