using Test

using AEPolytope
const AEP = AEPolytope

using Polymake

@test size(AEP.symmetrize(AEP.rand_sphere(3, 10)')) == (20, 3)

@testset "Number of vertices" begin
    @test AEP.integrate(0.0:0.01:1.0, x->x^2) ≈ 1/3
    @test AEP.integrate(0.0:0.01:1, cos) ≈ sin(1)
    @test AEP.integrate(-π:0.001:π, sin) ≈ 0.0 atol=1e-7

    @test AEP.normalized_cap_area(5, 1) ≈ 0.5
    @test AEP.normalized_cap_area(3, 0.5) ≈ 0.25

    @test AEP.hausdorff_dist_to_sphere(12, 160, 0.1) ≈ 0.80 atol=0.01
    @test AEP.hausdorff_dist_to_sphere( 5,  30, 0.05) ≈ 0.95 atol=0.001
end

@testset "scaling distance" begin
    A = let c = Polytope.cube(2, 1//4, -1//2)
        facets = @pm Common.convert_to{Matrix{Rational}}(c.FACETS)
        P = @pm Polytope.Polytope("POINTS"=>c.VERTICES, "INEQUALITIES"=>facets)
        Matrix{Float64}(@pm Common.convert_to{Matrix{Rational}}(P.INEQUALITIES))
    end

    pt = Float64[0, 1]
    @test AEP.scaling_distance(A, pt, [0,0])[1] == 4.0
    @test !in(A, pt, [0,0])

    pt = Float64[-2, 0]
    @test AEP.scaling_distance(A, pt, [0,0])[1] == 4.0
    @test !in(A, pt, [0,0])

    pt = Float64[0.25, -0.125]
    @test AEP.scaling_distance(A, pt, [0,0])[1] == 1.0
    @test in(A, pt, [0,0])

    pt = Float64[-0.25, -0.125]
    @test AEP.scaling_distance(A, pt, [0,0])[1] == 0.5
    @test in(A, pt, [0,0])

    @time AEP.scaling_distance(A, pt, [0,0]);
    @time in(A, pt, [0,0]);
#     0.000002 seconds (14 allocations: 816 bytes)
end
