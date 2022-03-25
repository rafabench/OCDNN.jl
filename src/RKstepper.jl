function RKstepper(Ctrls, rkmethod, Problem, c, mgrit)

    last = Ctrls.nlayers+1;
    if mgrit
        S=MGRITforwardstepper(Ctrls,rkmethod,Problem,c)
    else
        S=RKforwardstepper(Ctrls,rkmethod, Problem); #Step forward
    end
    S.P[last] = Problem.DJY(S.Y[last],Ctrls.W,Ctrls.mu,Problem.C)  #Set right boundary value for P
    if mgrit
        S=MGRITbackwardstepper(Ctrls,rkmethod, S,Problem,c)
    else
        S=RKbackwardstepper(Ctrls, rkmethod, S, Problem); # Step backwards to obtain the Ps
    end
    return S

end
