function RKstepper(Ctrls, rkmethod, C)

    last = Ctrls.nlayers+1;
    S=RKforwardstepper(Ctrls,rkmethod,C); #Step forward
    S.P[last] = DJY(S.Y[last],Ctrls.W,Ctrls.mu,C); #Set right boundary value for P
    S=RKbackwardstepper(Ctrls, rkmethod, C, S); # Step backwards to obtain the Ps
    return S

end
