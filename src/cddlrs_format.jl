function cddlrs_header(size::Tuple{Integer,Integer}, representation; elem_type=:rational, name="P")
    @assert representation in (:H, :V)
    @assert elem_type in (:rational, :integer)
    rows = [
        name,
        "# file written from julia",
        "$representation-representation",
        "begin",
        "$(size[1]) $(size[2]) $elem_type",
    ]
    return join(rows, "\n")
end

function cddlrs_format(M::Polymake.pm_Matrix)
    str = string(M)
    k = findfirst("\n", str)
    return strip(str[last(k)+1:end])
end

function cddlrs_format(M::Matrix)
    io = IOBuffer()
    Base.print_array(io, M)
    str = String(take!(io))
    return str
end

function cddlrs_format(M::Matrix{<:Rational})
    io = IOBuffer()
    Base.print_array(io, M)
    str = String(take!(io))
    str = replace(str, "//"=>"/")
    return str
end

function cddlrs_format(M::AbstractMatrix{T}, representation;
        elem_type=(T<:Integer ? :integer : :rational) , name="P") where T
    @assert representation in (:H, :V)
    @assert elem_type in (:integer, :rational)
    footer = "end"
    return join([
            cddlrs_header(size(M), representation, elem_type=elem_type, name=name),
            cddlrs_format(M),
            footer,
        ], "\n")
end

function cddlrs_format(polytope::pm_perl_Object, representation=:H; name="P")
    if representation == :H
        pts_to_write = P.INEQUALITIES
    elseif representation == :V
        pts_to_write = P.VERTICES
    else
        throw(ArgumentError("representation must be either :H or :V"))
    end

    return lrs_input(pts_to_write, representation, name=name)
end
