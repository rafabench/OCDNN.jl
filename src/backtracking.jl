function calculate_norm(Gradient, nlayers)
    normGsq = sum( Gradient.W .^2 ) + sum(Gradient.mu.^2);
    normGsq += sum( sum(Gradient.K[k] .^ 2) + Gradient.b[k]'*Gradient.b[k] for k in 1:nlayers);
    return normGsq
end

function backtracking(Ctrls,Gradient,rkmethod,Problem,τ, c1, mgrit)
    
    ρ = 0.5
    c = 0.4
    α = τ/ρ
    nlayers = Ctrls.nlayers;

    fk = objective(Ctrls,rkmethod,Problem, c1, mgrit)
    normGsq = calculate_norm(Gradient, nlayers);
    fnew = 2*fk;
    C1 = deepcopy(Ctrls)
    # Armijo's Condition
    while fnew > fk - α*c*normGsq
        α = ρ*α
        for k = 1:nlayers
            C1.K[k] = Ctrls.K[k] - α*Gradient.K[k];
            C1.b[k] = Ctrls.b[k] - α*Gradient.b[k];
        end
        C1.W = Ctrls.W - α*Gradient.W;
        C1.mu = Ctrls.mu - α*Gradient.mu;
        fnew = objective(C1,rkmethod,Problem, c1, mgrit);
    end
    return C1,α,normGsq
end

function GD(Ctrls,Gradient,rkmethod,Problem,τ, c1, mgrit)
    
    ρ = 0.5
    c = 0.4
    α = τ
    nlayers = Ctrls.nlayers;

    fk = objective(Ctrls,rkmethod,Problem, c1, mgrit)
    normGsq = calculate_norm(Gradient, nlayers);
    fnew = 2*fk;
    C1 = deepcopy(Ctrls)
    # Armijo's Condition
    for k = 1:nlayers
        C1.K[k] = Ctrls.K[k] - α*Gradient.K[k];
        C1.b[k] = Ctrls.b[k] - α*Gradient.b[k];
    end
    C1.W = Ctrls.W - α*Gradient.W;
    C1.mu = Ctrls.mu - α*Gradient.mu;
    return C1,α,normGsq
end
