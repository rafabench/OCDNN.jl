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

A_eu = reshape([0],(1,1));
w_eu = [1];
A_ie = [0 0; 1 0];
w_ie = [0.5 ;0.5];
A_kutta3 = [0 0 0;1/2 0 0;-1 2 0];
w_kutta3 = [1/6;2/3;1/6];
A_kutta4 = [0 0 0 0;0.5 0 0 0;0 0.5 0 0;0 0 1 0];
w_kutta4 = 1/6*[1;2;2;1];
A_kutta5 = [0 0 0 0 0; 2/5 0 0 0 0; 1/10 1/2 0 0 0; 1/16 1/16 3/8 0 0; 1/10 1/10 4/15 8/15 0]
w_kutta5 = [5/32;25/96;25/96;1/6;5/32];

RK5 = ExplicitRungeKutta(A_kutta5,w_kutta5,"RK5");
RK4 = ExplicitRungeKutta(A_kutta4,w_kutta4,"RK4");
RK3 = ExplicitRungeKutta(A_kutta3,w_kutta3,"RK3");
RK2 = ExplicitRungeKutta(A_ie,w_ie,"RK2");
RK1 = ExplicitRungeKutta(A_eu,w_eu,"Euler");