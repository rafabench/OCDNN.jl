function objective(Ctrls, rkmethod, Problem)
    #
    # Compute the objective function   F=1/2 * || eta(YN*W+mu) - C ||^2
    #
    # The RKforwardstepper obtains YN and the classifier eta(YN*W+mu)
    S1 = RKforwardstepper(Ctrls,rkmethod,Problem);
    return 1/2 * sum( (S1.Classifier - Problem.C).^2 );
end