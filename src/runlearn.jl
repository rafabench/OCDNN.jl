function train_ode_dnn(dataset, nlayers, channels, rkmethod, niter, τ, τ_max, stepsize)
    F = zeros(niter)
    Fn = zeros(niter)

    iter = 0
    α = τ/2

    Ctrls = ControlDef(dataset, stepsize, channels, nlayers);
    Y0, C = dataset
    for iter in 1:niter
        Gradient=GradientCalc(Ctrls, rkmethod, C);
        Ctrls,α,normGsq = backtracking(Ctrls,Gradient,rkmethod,C,min([τ_max,2*α]));
        F[iter] = objective(Ctrls, Method, C);
        Fn[iter] = sqrt(normGsq);
        if rem(iter,2000) == 0 || iter == 1
            printf("\nMethod=%s, nlayers=%d\n",rkmethod.name,nlayers)
            printf("Iteration no: %d\n",iter)
            printf("Value of alpha: %7.4f\n",alpha);
            printf("Residual: %8.4f\n",F[iter])
            printf("Norm Grad: %8.4e\n",Fn[iter]);
        end
    end
    return Ctrls,F,Fn
end
