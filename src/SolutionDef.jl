mutable struct SolutionDef
    channels::Int64 # Number of channels at least equal to cols
    rows::Int64  # Number of data points
    cols::Int64  # Degrees of freedom in each initial data point, col dim of Y0
    nlayers::Int64 # Number of steps taken by the integrator
    stepsize::Float64 # Stepsize of integrator
    Y::Array{Array{Float64,2},1} # Cell array (nlayers+1,1) holding the Y-values for all steps
    Ys::Array{Array{Float64,2},2} # Cell array (nlayers,nstages) holding all stage values for all steps
    P::Array{Array{Float64,2},1}  # Cell array - dual solution (nlayers+1,1) for all steps
    fPs::Array{Array{Float64,2},2} #  Cell array (nlayers,nstages) - stages of dual solution (derivatives)
    Classifier::Array{Float64,1}
    function SolutionDef(stepsize, nstages, Y0, channels, nlayers)
        N,d = size(Y0);
        rows = N;
        cols = d;
        Y = [randn(N,channels) for i = 1:nlayers+1];
        Y[1] = [Y0 zeros(N,channels-d)];
        Ys = [randn(N,channels) for i = 1:nlayers+1, j = 1:nstages];
        P = [randn(N,channels) for i = 1:nlayers+1];
        fPs = [randn(N,channels) for i = 1:nlayers+1, j = 1:nstages];
        Classifier = zeros(N)
        return new(channels,rows,cols,nlayers,stepsize,Y,Ys,P,fPs,Classifier)
    end
end