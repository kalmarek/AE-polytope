function randvert(ineqs::AbstractMatrix{<:Union{pm_Rational, Rational}}, n::Integer; range=-1_000_000:1_000_000)
    dim = size(ineqs, 2)

    obj = Polymake.pm_Vector{Polymake.pm_Rational}(dim)
    obj_int = Matrix{Int}(undef, dim, 2)
    verts = pm_Matrix{pm_Rational}(n, dim)

    for i in 1:n
        rand!(obj_int, range)
        obj_int[1,1] = 0
        obj .= obj_int[:,1].//(abs.(obj_int[:,2]) .+ 1)
        verts[i, :] .= Polymake.solve_LP(ineqs, obj)
    end
    return verts
end

function randvert(ineqs::AbstractMatrix{<:Union{pm_Rational, Rational}}; range=-1_000_000:1_000_000)
    obj = rand(range, size(ineqs, 2), 2)
    obj[1,1] = 0
    v = Polymake.solve_LP(ineqs, obj[:,1].//(abs.(obj[:,2]) .+ 1))
    return v
end

function randvert(ineqs::AbstractMatrix{<:AbstractFloat}, n::Integer)
    dim = size(ineqs, 2)

    obj = Polymake.pm_Vector{Float64}(dim)
    verts = pm_Matrix{Float64}(n, dim)

    v = Polymake.pm_Vector{Float64}(dim)
    k = 1;
    while k <= n
        rand!(obj)
        obj[1] = 0
        v = Polymake.solve_LP(ineqs, obj)
        if length(v) == dim
            verts[k, :] .= v
            k += 1
        end
    end
    return verts
end

function randvert(ineqs::AbstractMatrix{<:AbstractFloat})
    obj = rand(size(ineqs, 2))
    obj[1] = 0
    return Polymake.solve_LP(ineqs, obj)
end

randvert(P::Polymake.pm_perl_ObjectAllocated) = randvert(P.INEQUALITIES)

randvert(P::Polymake.pm_perl_ObjectAllocated, n::Integer) = randvert(P.INEQUALITIES, n)
