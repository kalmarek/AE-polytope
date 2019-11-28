end






function dual_bounding_body(pts::AbstractMatrix; hyperplanes=2*(size(pts,2)+1), k=1, seed=1234, digits=15)
    @assert size(pts, 2) <= 16 "Did You forgot to take transpose of pts?"

    P = Polytope.rand_sphere(size(pts,2), hyperplanes, seed=seed, precision=digits)
    verts = dehomogenize(Matrix{Float64}(P.VERTICES))

    # verts = symmetrize(rand_sphere(size(pts,2), hyperplanes, seed=seed, digits=digits)')

    mins = [-sort([dot(pts[i, :], verts[j, :]) for i in 1:size(pts, 1)])[k] for j in 1:size(verts, 1)]
    ineqs = rationalize.(augment(verts, mins))
    P = @pm Polytope.Polytope(INEQUALITIES=ineqs)
    return P
end

mutable struct RandomPolytopeClassifier
    polytope::Polymake.pm_perl_ObjectAllocated
    inequalities::Matrix{Float64}
    center::Vector{Float64}
    diameter::Float64
end

function RandomPolytopeClassifier(pts::AbstractMatrix;
        hyperplanes=min(50, 2*(size(pts,2)+1)),
        k=2, kwargs...)

    P = dual_bounding_body(pts', hyperplanes=hyperplanes, k=k; kwargs...)
    verts = Matrix{Float64}(P.VERTICES)

    ineqs = Matrix{Float64}(P.INEQUALITIES)
    center = (sum(verts, dims=1)./size(verts, 1))[1,:]

    rpc = RandomPolytopeClassifier(P, ineqs, dehomogenize(center),
            0.0) # placeholder for Float64(Polytope.diameter(P))

    return rpc
end

function Base.in(point, rpc::RandomPolytopeClassifier)
    return in(rpc.inequalities, point, rpc.center)
end

function distance(rpc::RandomPolytopeClassifier, pt::AbstractVector)
    return max(0.0, scaling_distance(rpc.inequalities, pt, rpc.center)[1])
end

score(rpc::RandomPolytopeClassifier, point) = distance(rpc, point)

function scale_verts(P::RandomPolytopeClassifier, scale)
    points = dehomogenize(Matrix{Float64}(P.polytope.VERTICES))
    points = scale!(points, P.center, scale)
    return points
end
