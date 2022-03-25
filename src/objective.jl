function objective(Ctrls, rkmethod, Problem, c, mgrit)
    #
    # Compute the objective function   F=1/2 * || eta(YN*W+mu) - C ||^2
    #
    # The RKforwardstepper obtains YN and the classifier eta(YN*W+mu)
    if mgrit
        S1=MGRITforwardstepper(Ctrls,rkmethod,Problem,c)
    else
        S1=RKforwardstepper(Ctrls,rkmethod, Problem); #Step forward
    end
    return 1/2 * sum( (S1.Classifier - Problem.C).^2 );
end