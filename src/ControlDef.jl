mutable struct ControlDef
    Y0::Array{Float64,2} # Matrix (N,channels) of initial data
    channels::Int64 # Number of channels at least equal to cols
    rows::Int64  # Number of data points
    cols::Int64  # Degrees of freedom in each initial data point, col dim of Y0
    nlayers::Int64 # Number of steps taken by the integrator (elements of cell arrays K and b)
    K::Array{Float64,3} # Cell array of matrices containing control parameters
    b::Array{Float64,2} # Cell array of row vectors containing control biases
    W::Array{Float64,1} # Array (channels,1) containing the control parameters of the final projection
    mu::Float64 # scalar bias for final projection (classifier)
    stepsize::Float64
    function ControlDef(dataset, stepsize, channels, nlayers)
        Y0, C = dataset
        N,d = size(dataset)
        Y0 = [Y0 zeros(N,channels-d)]
        rows = N
        cols = d
        K = randn(nlayers,channels,channels)
        b = randn(nlayers,channels)
        W = randn(channels)
        mu = randn(1)
        return new(Y0,channels,rows,cols,nlayers,K,b,W,mu,stepsize)
    end
end

