mutable struct ExplicitRungeKutta
    s::Int64
    A::Array{Float64,2}
    w::Array{Float64,1}
    At::Array{Float64,2}
    name::String
    function ExplicitRungeKutta(A,w,name="Euler")
        s = size(A,1);
        if s <= 0
            error("number of stages must be positive")
        end
        if norm(UpperTriangular(A)) > 0
            error("method must be explicit")
        end
        if minimum(abs.(w))==0
            error("all weights must be non-zero")
        end
        At = zeros(size(A))
        for i = 1:s, j = 1:s
            At[i,j] = w[j]/w[i]*A[j,i];
        end
        return new(s,A,w,At,name)
    end
end