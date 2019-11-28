import Base.in
for f in [:scaling_distance, :in]
    @eval begin
        function $f(A::AbstractMatrix{T}, pt::AbstractVector, center::AbstractVector) where T
        #     we will be looking at intersection of
        #     ray   t·v + c, t≥0, and
        #     plane a·x - b = 0, x ∈ R^n

        #     @assert length(pt) == length(center) == size(A,2) - 1

            @assert size(A, 2) - 1 == size(pt, 1) == size(center, 1)

            v = pt - center
            a = Vector{T}(undef, length(v))
            idx = 0
            scale = T(1000)

            # in A each row corresponds to [b, -a], i.e. b - a₁x₁ - ... -aₙxₙ ≥ 0
            # A corresponds to inward normal vectors:

            for i in 1:size(A, 1)
                b = A[i, 1]
                for j in 2:size(A, 2)
                    a[j-1] = -A[i, j] # no allocation copy
                end

                av = dot(a,v)
                if av > T(1e-3) # we expect scale < 1000.0
                    t = (b - dot(a, center))/av
                    @debug "a,b,av,t" a b av t
                    if T(0) < t < scale
                        scale = t
                        idx = i
                        @debug "scale" scale
                        if Symbol($f) == :in
                            scale < 1.0 && return false
                        end
                    end
                end
            end
            if Symbol($f) == :in
                return true
            else
                return inv(scale), idx
            end
        end
    end
end
