using Test

using Polymake

@testset "scaling distance" begin
    A = let c = Polytope.cube(2, 1//4, -1//2)
        facets = @pm Common.convert_to{Matrix{Rational}}(c.FACETS)
        P = @pm Polytope.Polytope("POINTS"=>c.VERTICES, "INEQUALITIES"=>facets)
        Matrix{Float64}(@pm Common.convert_to{Matrix{Rational}}(P.INEQUALITIES))
    end

    pt = Float64[0, 1]
    @test AE.scaling_distance(A, pt, [0,0])[1] == 4.0
    @test !in(A, pt, [0,0])

    pt = Float64[-2, 0]
    @test AE.scaling_distance(A, pt, [0,0])[1] == 4.0
    @test !in(A, pt, [0,0])

    pt = Float64[0.25, -0.125]
    @test AE.scaling_distance(A, pt, [0,0])[1] == 1.0
    @test in(A, pt, [0,0])

    pt = Float64[-0.25, -0.125]
    @test AE.scaling_distance(A, pt, [0,0])[1] == 0.5
    @test in(A, pt, [0,0])

    @time AE.scaling_distance(A, pt, [0,0]);
    @time in(A, pt, [0,0]);
#     0.000002 seconds (14 allocations: 816 bytes)
end
