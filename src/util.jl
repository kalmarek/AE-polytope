using Test
max_kth!(perm::AbstractVector{<:Integer}, itr, k::Integer) = itr[partialsortperm!(perm, itr, k, rev=true)]

using DelimitedFiles

function symmetrize(pts::AbstractMatrix, augmented=false)
    V = vcat(pts, -pts)
    if augmented
        V[size(pts,1)+1:end, 1] .= one(eltype(V))
    end
    return V
end

symmetrize(poly) = @pm Polytope.Polytope(POINTS=symmetrize(poly.VERTICES, true))

function rand_sphere(dim, no_points; seed=rand(1:1000), digits=15) where T
    Random.seed!(seed)
    pts = randn(dim, no_points)
    for i in 1:size(pts, 2)
        pts[:, i] ./= norm(pts[:, i], 2)
    end
    return round.(pts, digits=digits)
end

@test size(symmetrize(rand_sphere(3, 10)')) == (20, 3)

function scale!(points, center, scale)
    for i in 1:size(points, 1)
        @views points[i, :] .= scale.*(points[i, :] .- center) .+ center
    end
    return points
end
