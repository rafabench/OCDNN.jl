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

function MGRITbackwardstepper(Ctrls,rkmethod, S,Problem,c)
   h = Ctrls.stepsize;
   last = Ctrls.nlayers+1;
   S1 = deepcopy(S);

   nlayers = Ctrls.nlayers;
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
   # Backward stepping
   for k = coerse_nlayers-1:-1:1
      n = k*c+1
      Ps = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
      for i = rkmethod.s:-1:1
         Ps[i] = S1.P[n];
         for j = i+1:rkmethod.s
            Ps[i] = Ps[i]-c*h*rkmethod.At[i,j]*S1.fPs[n-c,j];
         end
         S1.fPs[n-c,i] = Problem.AdjVf(S1.Ys[n-1,i],Ps[i],Ctrls.K[n-c],Ctrls.b[n-c]);
      end
      S1.P[n-c] = S1.P[n];
      for i = 1:rkmethod.s
         S1.P[n-c] = S1.P[n-c]-c*h*rkmethod.w[i]*S1.fPs[n-c,i];
      end
   end
   while true
      ##################
      # FCF Relaxation #
      ##################

      # F-relaxation
      @threads for k = coerse_nlayers-2:-1:0
         for n = (k+1)*c+1:-1:k*c+3
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
      end

      # C-relaxation
      @threads for k = coerse_nlayers-2:-1:0
         n = k*c+2
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

      # F-relaxation
      @threads for k = coerse_nlayers-2:-1:0
         for n = (k+1)*c+1:-1:k*c+3
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
      end

      # Calculate Residual
      for k = coerse_nlayers-2:-1:0
         n = k*c+2
         U[k+1] = S1.P[n];
         U_C[k+1] = U[k+1]
         Ps = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
         for i = rkmethod.s:-1:1
            Ps[i] = S1.P[n];
            for j = i+1:rkmethod.s
               Ps[i] = Ps[i]-h*rkmethod.At[i,j]*S1.fPs[n-1,j];
            end
            S1.fPs[n-1,i] = Problem.AdjVf(S1.Ys[n-1,i],Ps[i],Ctrls.K[n-1],Ctrls.b[n-1]);
         end
         for i = 1:rkmethod.s
            U_C[k+1] = U_C[k+1]-h*rkmethod.w[i]*S1.fPs[n-1,i];
         end
         R[k+1] = - (S1.P[n-1] - U_C[k+1])
     end

     for k = coerse_nlayers-2:-1:0
         n = (k+1)*c+1
         U[k+2] = S1.P[n];
         Ps = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
         for i = 1:rkmethod.s
            Ps[i] = S1.P[n];
            for j = 1:i-1
               Ps[i] = Ps[i]-c*h*rkmethod.At[i,j]*S1.fPs[n-c,j];
            end
            S1.fPs[n-c,i] = Problem.AdjVf(S1.Ys[n-c,i],Ps[i],Ctrls.K[n-c],Ctrls.b[n-c]);
         end
         U[k+1] = U[k+2];
         for i = 1:rkmethod.s
            U[k+1]=U[k+1]-c*h*rkmethod.w[i]*S1.fPs[n-c,i];
         end
         A_U[k+1] =  S1.P[n-c] - U[k+1]
      end

      V[coerse_nlayers] = U[coerse_nlayers]
      for k = coerse_nlayers-2:-1:0
         n = (k+1)*c+1
         Ps = [randn(size(Ctrls.Y0)) for i = 1:rkmethod.s];
         for i = 1:rkmethod.s
            Ps[i] = S1.P[n];
            for j = 1:i-1
               Ps[i] = Ps[i]-c*h*rkmethod.At[i,j]*S1.fPs[n-c,j];
            end
            S1.fPs[n-c,i] = Problem.AdjVf(S1.Ys[n-c,i],Ps[i],Ctrls.K[n-c],Ctrls.b[n-c]);
         end
         V[k+1] = V[k+2];
         for i = 1:rkmethod.s
            V[k+1]=V[k+1]-c*h*rkmethod.w[i]*S1.fPs[n-c,i];
         end
         V[k+1] += R[k+1] + A_U[k+1]
      end

      # Compute coerse grid error approximation
      E = V .- U

      # Correct grid
      for k = coerse_nlayers-2:-1:0
         n = k*c+1
         S1.P[n] += E[k+1]
      end

      # F-relaxation
      @threads for k = coerse_nlayers-2:-1:0
         for n = (k+1)*c+1:-1:k*c+3
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
      end

      iter += 1
      if norm(R) < 1e-5 || iter > maxiter
         break
      end
   end

   return S1
end