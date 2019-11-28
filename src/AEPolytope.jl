module AEPolytope

using Random
using Statistics
using LinearAlgebra
using Polymake
import OscarPolytope: augment, homogenize, dehomogenize

export RandomPolytopeClassifier, dual_bounding_body, randvert

include("util.jl")
include("scaling_distance.jl")
include("rpc.jl")
include("randvert.jl")

include("data_wrangling.jl")
# include("visualisation.jl")

end # of module
