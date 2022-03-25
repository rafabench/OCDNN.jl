__precompile__()

module OCDNN

    using Random, LinearAlgebra, ForwardDiff, Printf
    using Base.Threads

    include("build_datasets.jl")
    include("HBVPdef.jl")
    include("ControlDef.jl")
    include("GradientCalc.jl")
    include("backtracking.jl")
    include("objective.jl")
    include("SolutionDef.jl")
    include("RKbackwardstepper.jl")
    include("RKforwardstepper.jl")
    include("RKstepper.jl")
    include("ExplicitRungeKutta.jl")
    include("runlearn.jl")

    export build_dataset, train_ode_dnn, circle, halfspace, four_regions, 
    Problem, RKforwardstepper, ExplicitRungeKutta, SolutionDef, RK1, RK2, RK3, RK4, RK5

end
