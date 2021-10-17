using Random

Random.seed!(1234);

function build_dataset(f = circle; N_samples = 1000, N_noise = 50)
    data = rand(N_samples,2)
    feat = f.(eachslice(data,dims=2)...)
    # Change label of random indices
    idxs_noised = rand(1:N_samples,N_noise)
    feat[idxs_noised] = .!feat[idxs_noised]
    return data, feat
end

circle(x,y) = (x-0.5)^2+(y-0.5)^2 <= 0.3^2
halfspace(x,y) = x + y <= 1
four_regions(x,y) = sign(x-0.5)*sign(y-0.5) <= 0