mutable struct ControlDef
    Y0::Array{Float64,2} # Matrix (N,channels) of initial data
    channels::Int64 # Number of channels at least equal to cols
    rows::Int64  # Number of data points
    cols::Int64  # Degrees of freedom in each initial data point, col dim of Y0
    nlayers::Int64 # Number of steps taken by the integrator (elements of cell arrays K and b)
    K::Array{Array{Float64,2},1} # Cell array of matrices containing control parameters
    b::Array{Array{Float64,1},1} # Cell array of row vectors containing control biases
    W::Array{Float64,1} # Array (channels,1) containing the control parameters of the final projection
    mu::Float64 # scalar bias for final projection (classifier)
    stepsize::Float64
    function ControlDef(dataset, stepsize, channels, nlayers)
        Y0, C = dataset
        N,d = size(dataset)
        Y0 = [Y0 zeros(N,channels-d)]
        rows = N
        cols = d
        K = [randn(channels,channels) for i in 1:nlayers]
        b = [randn(channels) for i in 1:nlayers]
        W = randn(channels)
        mu = randn()
        return new(Y0,channels,rows,cols,nlayers,K,b,W,mu,stepsize)
    end
end

