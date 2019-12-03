mutable struct RandomPolytopeClassifier
    polytope::Polymake.pm_perl_ObjectAllocated
    inequalities::Matrix{Float64}
    center::Vector{Float64}
    diameter::Float64
end

function dual_bounding_body(pts::AbstractMatrix{T}, n_hyperplanes::Integer, k::Integer=1; seed=1234, digits=15) where T

    @assert size(pts, 2) <= 100 "Did You forgot to take transpose of pts?"
    # P = Polytope.rand_sphere(size(pts,2), hyperplanes, seed=seed, precision=digits)
    # verts = dehomogenize(Matrix{Float64}(P.VERTICES))

    verts = symmetrize(rand_sphere(size(pts,2), n_hyperplanes, seed=seed, digits=digits)') # Matrix{Float64}

    # mins = @views [
        # maxk([dot(pts[i, :], verts[j, :]) for i in 1:size(pts, 1)], k)
        # for j in 1:size(verts, 1)
        # ]

    n_hyperplanes = size(verts, 1)
    S = promote_type(T, eltype(verts))

    dots = Vector{S}(undef, size(pts,1))
    mins = Vector{S}(undef, n_hyperplanes)

    sort_perm = Vector{Int}(undef, size(dots, 1)) # temporary storage

    for j in eachindex(mins)
        for i in 1:size(pts, 1)
            dots[i] = @views -dot(pts[i, :], verts[j, :])
        end
        mins[j] = max_kth!(sort_perm, dots, k)
    end

    ineqs = rationalize.(augment(verts, mins), tol=10.0^-digits)
    P = @pm Polytope.Polytope(INEQUALITIES=ineqs)
    return P
end

function RandomPolytopeClassifier(points::AbstractMatrix, n_hyperplanes;
        k=2, p=0.05, ε=0.1, seed=1234, digits=10)

    P = dual_bounding_body(points', n_hyperplanes, k; seed=seed, digits=digits)
    n_hyperplanes = size(P.INEQUALITIES, 1)

    dim = size(points, 1)
    dist_to_sphere = hausdorff_dist_to_sphere(dim, n_hyperplanes, p)

    N = number_of_vertices(dim, dist_to_sphere; p=p, ε=ε)
    # @info "RandomPolytopeClassifier:" dim dist_to_sphere number_of_vertices=N

    verts = Matrix{Float64}(randvert(@pm(Common.convert_to{Float}(P)), N))
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
