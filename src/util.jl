min_kth!(perm::AbstractVector{<:Integer}, itr, k::Integer) = itr[partialsortperm!(perm, itr, k, rev=false)]

function symmetrize(pts::AbstractMatrix, augmented::Bool=false)
    V = vcat(pts, -pts)
    if augmented
        V[size(pts,1)+1:end, 1] .= one(eltype(V))
    end
    return V
end

symmetrize(poly::Polymake.pm_perl_Object) = @pm Polytope.Polytope(POINTS=symmetrize(poly.VERTICES, true))

function rand_sphere(dim::Integer, no_points::Integer) where T
    pts = randn(dim, no_points)
    for i in 1:size(pts, 2)
        pts[:, i] ./= norm(pts[:, i], 2)
    end
    return pts
end

function scale!(points::AbstractMatrix, center::AbstractVector, scale::Number)
    for i in 1:size(points, 1)
        @views points[i, :] .= scale.*(points[i, :] .- center) .+ center
    end
    return points
end

#######################
# Integration and volume of the ball (needed for estimate of the number of points).

using SpecialFunctions: gamma

ball_volume(dim, radius=1.0) = sqrt(π)^dim / gamma(dim/2+1)*radius^dim

struct Simpson end
integrate(::Type{Simpson}, a::Number, b::Number, f) =
    (b-a)/6*(f(a) + 4f((a+b)/2) + f(b))
integrate(a::Number, b::Number, f) = integrate(Simpson, a, b, f)

function integrate(range::AbstractVector, f)
    first(range) == last(range) && return zero(eltype(range))
    return sum(integrate(a, b, f) for (a,b) in zip(range, Iterators.rest(range, 2)))
end

function normalized_cap_area(dim, height)
    iszero(height) && return zero(height);
    a = sqrt(2height-height^2)^(dim-1)*(1.0-height)/dim
    b = integrate(1-height:height/100:1, y-> (1-y^2)^((dim-1)/2))
    return (a+b)*ball_volume(dim-1)/ball_volume(dim)
end

function hausdorff_dist_to_sphere(dim::Integer, n_hyperplanes::Integer, p::Number; atol=10^3*eps(p))
    up = one(p)
    lo = zero(p)
    m_choose_d = Float64(binomial(big(n_hyperplanes), dim))
    diff = (up - lo)/2
    while diff > atol
        mid = (up + lo)/2
        val = m_choose_d*(1 - normalized_cap_area(dim, mid))^(n_hyperplanes-dim)
        @debug up mid lo val diff
        if val > p/2
            lo = mid
        else
            up = mid
        end
        diff = (up - lo)/2
    end
    return (up + lo)/2
end

function number_of_vertices(dim::Integer, hausdorff_dist::Number; p=0.05, ε=0.1)
    n = (1 + 2/dim*log(2/p))*(ℯ/(ℯ-1))*1/(ε^2*(1-hausdorff_dist)^2)
    n > typemax(Int) && throw("Number of vertices exceeds $(typemax(Int)). Try increasing Hausdorff distance, p, or ε!")
    return ceil(Int, n)
end
