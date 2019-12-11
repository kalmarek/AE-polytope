mutable struct RandomPolytopeClassifier
    polytope::Polymake.pm_perl_ObjectAllocated
    inequalities::Matrix{Float64}
    center::Vector{Float64}
    diameter::Float64
end

function mydot(x::AbstractVector{T}, y::AbstractVector{S}) where {T,S}
    @boundscheck length(x) == length(y)
    s = zero(promote_type(T, S))
    @inbounds for i in 1:length(x)
        s += x[i]*y[i]
    end
    return s
end

function dual_inequalities(pts::AbstractArray, n_hyperplanes::Integer, k::Integer=1)
    dim = size(pts, 1)

    @assert dim <= 50 "Dimensionality of the data is too large for feasible computations: dim = $dim. Try with dim <= 50."

    # P = Polytope.rand_sphere(size(pts,2), hyperplanes, seed=seed, precision=digits)
    # verts = dehomogenize(Matrix{Float64}(P.VERTICES))

    ineqs = symmetrize(rand_sphere(dim, n_hyperplanes)') # Matrix{Float64}
    #NOTE: ineqs are ~ of the size (2n_hyperplanes, dim), i.e. it's TRANSPOSED
    S = promote_type(eltype(pts), eltype(ineqs))
    intercepts = Vector{S}(undef, size(ineqs, 1))

    dots = Vector{S}(undef, size(pts, 2))
    # temporary storage for sorting dots
    sort_perm = Vector{Int}(undef, size(dots, 1))

    @inbounds @views for j in 1:size(ineqs, 1)
        for i in 1:size(pts, 2)
            dots[i] = mydot(pts[:, i], ineqs[j, :])
        end
        intercepts[j] = min_kth!(sort_perm, dots, k)
    end

    return ineqs, intercepts # describes ineqs*x + intercepts > 0
end

function dual_bounding_body(pts::AbstractMatrix, n_hyperplanes::Integer, k::Integer=1; digits=10)

    @assert size(pts, 1) <= 50 "Did You forgot to take transpose of pts?"

    A, b = dual_inequalities(pts, n_hyperplanes, k)
    ineqs = rationalize.(augment(A, -b), tol=10.0^-digits)

    return @pm Polytope.Polytope(INEQUALITIES=ineqs) # Polytope{pm_Rational}
end

function dual_bounding_body(pts::AbstractMatrix{T}, n_hyperplanes::Integer, k::Integer=1; digits=10) where T <: AbstractFloat

    @assert size(pts, 1) <= 50 "Did You forgot to take transpose of pts?"

    A, b = dual_inequalities(pts, n_hyperplanes, k)
    ineqs = augment(A, -b)

    return @pm Polytope.Polytope{Float}(INEQUALITIES=ineqs) # Polytope{pm_Rational}
end

function RandomPolytopeClassifier(points::AbstractMatrix, n_hyperplanes;
        k=2, p=0.05, ε=0.1, digits=10)

    @time P = dual_bounding_body(points, n_hyperplanes, k; digits=digits)
    n_hyperplanes = size(P.INEQUALITIES, 1)

    dim = size(points, 1)
    dist_to_sphere = hausdorff_dist_to_sphere(dim, n_hyperplanes, p)

    N = number_of_vertices(dim, dist_to_sphere; p=p, ε=ε)
    # @info "RandomPolytopeClassifier:" dim dist_to_sphere number_of_vertices=N
    Pfl = @pm Common.convert_to{Float}(P)
    ineqs = Matrix{Float64}(Pfl.INEQUALITIES)
    @time verts = Matrix{Float64}(randvert(ineqs, N))
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
