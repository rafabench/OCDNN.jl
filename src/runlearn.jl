mutable struct Logger
    loss::Array{Float64,1}
    grad_norm::Array{Float64,1}
    accuracy::Array{Float64,1}
    time::Array{Float64,1}
    function Logger()
        loss = []
        grad_norm = []
        accuracy = []
        time = []
        new(loss,grad_norm,accuracy,time)
    end
end

function calculate_accuracy(Ctrls,rkmethod,Problem)
    S = RKforwardstepper(Ctrls,rkmethod,Problem)
    Classifier = S.Classifier
    acc = sum(round.(Classifier) .== Problem.C)/length(Problem.C)
    return acc
end

function train_ode_dnn(dataset, nlayers, channels, rkmethod, niter, τ, τ_max, stepsize, problem, cb)
    F = zeros(niter)
    Fn = zeros(niter)

    iter = 0
    α = τ/2

    Ctrls = ControlDef(dataset, stepsize, channels, nlayers);
    Y0, C = dataset
    log = Logger()
    for iter in 1:niter
        time1 = time_ns()
        Gradient=GradientCalc(Ctrls, rkmethod, problem);
        Ctrls,α,normGsq = backtracking(Ctrls,Gradient,rkmethod,problem,minimum([τ_max,2*α]));
        time2 = time_ns()
        Δt = (time2 - time1)/1e9
        push!(log.loss,objective(Ctrls, rkmethod, problem))
        push!(log.grad_norm,sqrt(normGsq))
        push!(log.accuracy,calculate_accuracy(Ctrls,rkmethod,problem))
        push!(log.time,Δt)

        cb(iter, rkmethod, nlayers, α, log, Ctrls);
    end
    return Ctrls,log
end
