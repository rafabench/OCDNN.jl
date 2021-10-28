__precompile__()

module OCDNN

    using Random

    include("build_datasets.jl")
    include("HBVPdef.jl")
    include("ControlDef.jl")
    include("GradientCalc.jl")
    include("backtracking.jl")
    include("objective.jl")
    include("runlearn.jl")

    export build_dataset, train_ode_dnn, circle, halfspace, four_regions

end
