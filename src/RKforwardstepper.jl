function RKforwardstepper(Ctrls,rkmethod,Problem)

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
           F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
        end
        S.Y[n+1] = S.Y[n];
        for i = 1:rkmethod.s
            S.Y[n+1]=S.Y[n+1]+h*rkmethod.w[i]*F[i];
        end
    end

    S.Classifier = Problem.eta.(S.Y[last]*Ctrls.W .+ Ctrls.mu');
    return S
end

function MGRITforwardstepper(Ctrls,rkmethod,Problem,c)
    nlayers = Ctrls.nlayers;
    channels = Ctrls.channels;
    h = Ctrls.stepsize;
    last = nlayers+1;

    S = SolutionDef(h, rkmethod.s, Ctrls.Y0, channels, nlayers);
    coerse_nlayers = Int64(nlayers/c) + 1
    U = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers]
    Us = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers, j = 1:rkmethod.s]
    V = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers]
    Vs = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers, j = 1:rkmethod.s]
    U_C = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers]
    R = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers]
    A_U = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers]
    E = [zeros(size(Ctrls.Y0)) for i = 1:coerse_nlayers]
    iter = 0
    maxiter = 5

    # Coerse Grid
    # Forward stepping
    F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
    for k = 1:coerse_nlayers-1
        n = (k-1)*c+1
        for i = 1:rkmethod.s
            S.Ys[n,i]=S.Y[n];
            for j = 1:i-1
                S.Ys[n,i]=S.Ys[n,i].+c*h*rkmethod.A[i,j]*F[j];
            end
            F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
        end
        S.Y[n+c] = S.Y[n];
        for i = 1:rkmethod.s
            S.Y[n+c]=S.Y[n+c]+c*h*rkmethod.w[i]*F[i];
        end
    end
    while true
        ##################
        # FCF Relaxation #
        ##################

        # F-relaxation
        F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
        @threads for k = 0:coerse_nlayers-2
            for n = k*c+1:(k+1)*c-1
                for i = 1:rkmethod.s
                    @inbounds S.Ys[n,i]=S.Y[n];
                    for j = 1:i-1
                        @inbounds S.Ys[n,i]=S.Ys[n,i].+h*rkmethod.A[i,j]*F[j];
                    end
                    @inbounds F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
                end
                @inbounds S.Y[n+1] = S.Y[n];
                for i = 1:rkmethod.s
                    @inbounds S.Y[n+1]=S.Y[n+1]+h*rkmethod.w[i]*F[i];
                end
            end
        end

        # C-relaxation
        F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
        @threads for k = 1:coerse_nlayers-1
            n = k*c
            for i = 1:rkmethod.s
                @inbounds S.Ys[n,i]=S.Y[n];
                for j = 1:i-1
                    @inbounds S.Ys[n,i]=S.Ys[n,i].+h*rkmethod.A[i,j]*F[j];
                end
                @inbounds F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
            end
            @inbounds S.Y[n+1] = S.Y[n];
            for i = 1:rkmethod.s
                @inbounds S.Y[n+1]=S.Y[n+1]+h*rkmethod.w[i]*F[i];
            end
        end

        # F-relaxation
        F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
        @threads for k = 0:coerse_nlayers-2
            for n = k*c+1:(k+1)*c-1
                for i = 1:rkmethod.s
                    @inbounds S.Ys[n,i]=S.Y[n];
                    for j = 1:i-1
                        @inbounds S.Ys[n,i]=S.Ys[n,i].+h*rkmethod.A[i,j]*F[j];
                    end
                    @inbounds F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
                end
                @inbounds S.Y[n+1] = S.Y[n];
                for i = 1:rkmethod.s
                    @inbounds S.Y[n+1]=S.Y[n+1]+h*rkmethod.w[i]*F[i];
                end
            end
        end

        # Calculate Residual
        for k = 1:coerse_nlayers-1
            n = k*c
            U[k+1] = S.Y[n]
            U_C[k+1] = U[k+1]
            for i = 1:rkmethod.s
                F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
                U_C[k+1] = U_C[k+1]+h*rkmethod.w[i]*F[i]
            end
            R[k+1] = - (S.Y[n+1] - U_C[k+1])
        end

        for k = 1:coerse_nlayers-1
            n = (k-1)*c+1
            U[k] = S.Y[n]
            for i = 1:rkmethod.s
                Us[k,i]=U[k];
                for j = 1:i-1
                    Us[k,i]=Us[k,i].+c*h*rkmethod.A[i,j]*F[j];
                end
                F[i] = Problem.Vf(Us[k,i],Ctrls.K[n],Ctrls.b[n])
            end
            U[k+1] = U[k];
            for i = 1:rkmethod.s
                U[k+1]=U[k+1]+c*h*rkmethod.w[i]*F[i];
            end
            A_U[k+1] =  S.Y[n+c] - U[k+1]
        end
        
        # Compute Coerse grid approximation 
        V[1] = U[1]
        F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
        for k = 0:coerse_nlayers-2
            n = k*c+1
            for i = 1:rkmethod.s
                Vs[k+1,i]=V[k+1];
                for j = 1:i-1
                    Vs[k+1,i]=Vs[k+1,i].+c*h*rkmethod.A[i,j]*F[j];
                end
                F[i] = Problem.Vf(Vs[k+1,i],Ctrls.K[n],Ctrls.b[n])
            end
            V[k+2] = V[k+1];
            for i = 1:rkmethod.s
                V[k+2]=V[k+2]+c*h*rkmethod.w[i]*F[i];
            end
            V[k+2] += R[k+2] + A_U[k+2]
        end

        # Compute coerse grid error approximation
        E = V .- U

        # Correct grid
        for k = 1:coerse_nlayers-1
            n = k*c+1
            S.Y[n] += E[k+1]
        end

        F = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
        @threads for k = 0:coerse_nlayers-2
            for n = k*c+1:(k+1)*c-1
                for i = 1:rkmethod.s
                    @inbounds S.Ys[n,i]=S.Y[n];
                    for j = 1:i-1
                        @inbounds S.Ys[n,i]=S.Ys[n,i].+h*rkmethod.A[i,j]*F[j];
                    end
                    @inbounds F[i] = Problem.Vf(S.Ys[n,i],Ctrls.K[n],Ctrls.b[n]);
                end
                @inbounds S.Y[n+1] = S.Y[n];
                for i = 1:rkmethod.s
                    @inbounds S.Y[n+1]=S.Y[n+1]+h*rkmethod.w[i]*F[i];
                end
            end
        end

        iter += 1
        if norm(R) < 1e-5 || iter > maxiter
            break
        end
    end
    
    S.Classifier = Problem.eta.(S.Y[last]*Ctrls.W .+ Ctrls.mu');
    return S
end
