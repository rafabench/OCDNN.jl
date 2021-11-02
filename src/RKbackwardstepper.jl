function RKbackwardstepper(Ctrls, rkmethod, S, Problem)

    # This function assumes that the input S contains the adjoint variable
    # S.P[nlayers+1] obtained after RKforwardstepper has been called
    h = Ctrls.stepsize;
    last = S.nlayers+1;
    S1 = deepcopy(S);
    # Backward stepping
    for n=last:-1:2
       Ps = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
       for i = rkmethod.s:-1:1
          Ps[i] = S1.P[n];
          for j = i+1:rkmethod.s
             Ps[i] = Ps[i]-h*rkmethod.At[i,j]*S1.fPs[n-1,j];
          end
          S1.fPs[n-1,i] = Problem.AdjVf(S1.Ys[n-1,i],Ps[i],Ctrls.K[n-1],Ctrls.b[n-1]);
       end
       S1.P[n-1] = S1.P[n];
       for i = 1:rkmethod.s
          S1.P[n-1] = S1.P[n-1]-h*rkmethod.w[i]*S1.fPs[n-1,i];
       end
    end
    return S1
end