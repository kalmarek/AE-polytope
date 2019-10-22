ENV["OMP_NUM_THREADS"] = 1
using ROC

using Random
using Statistics
using LinearAlgebra
using Polymake
import OscarPolytope: augment, homogenize, dehomogenize

include("../src/util.jl")
include("../src/data_wrangling.jl")
include("../src/rpc.jl")

using MLBase
using MLDatasets
using ImageCore

train_data, train_labels = MNIST.traindata(Float64);
train_features = MNIST.convert2features(train_data)
test_data,  test_labels = MNIST.testdata(Float64);
test_features = MNIST.convert2features(test_data);


function create_rpc_oneclass(features, labels, class; hyperplanes=60, k=2, seed=1, digits=5)

    (train_features, train_labels) = prepare_poisoned_data(features', labels, id=class, percentage_outliers=0.02)

    P = dual_bounding_body(train_features'; hyperplanes=hyperplanes, k=k, seed=seed, digits=digits)

    verts = Matrix{Float64}(P.VERTICES)
    ineqs = Matrix{Float64}(P.INEQUALITIES)

    center = (sum(verts, dims=1)./size(verts, 1))[1,:]
    rpc = RandomPolytopeClassifier(P, ineqs, dehomogenize(center), 0.0)

    train_labels = labels01(train_labels, class)
    @assert size(train_features, 2) == size(train_labels, 1)
    predicted_scores = [score(rpc, train_features[:, i]) for i in eachindex(train_labels)]
    roc_rpc = ROC.roc(predicted_scores, train_labels, distances=false);
#     @info "Obtained train accuracy for class=$class" auc=ROC.AUC(roc_rpc)
    return rpc, train_features, train_labels
end

function test_auc(train_features, train_labels, test_features, test_labels, class; kwargs...)
    rpc, _, _ = create_rpc_oneclass(train_features, train_labels, class; kwargs...)
    scores_test = [score(rpc, test_features[:, i]) for i in eachindex(test_labels)]
    labels_test = labels01(test_labels, class)[:, 1]

    r = ROC.roc(scores_test, labels_test, distances=false)

    return ROC.AUC(r)
end

const EXPERIMENTS_AE = Dict(
    6 => [joinpath(AE_prefix(), "log/mnist_embed/autoencoder/mnist/rep_dim=6/seed_$i") for i in 1:5],
    8 => [joinpath(AE_prefix(), "log/mnist_embed/autoencoder/mnist/rep_dim=8/seed_$i") for i in 1:5],
    12 => [joinpath(AE_prefix(), "log/mnist_embed/autoencoder/mnist/rep_dim=12/seed_$i") for i in 1:5]
);

const EXPERIMENTS_VAE = Dict(
    6 =>  [joinpath(AE_prefix(), "log/mnist_embed/vae/mnist/rep_dim=6/seed_$i") for i in 1:5],
    8 =>  [joinpath(AE_prefix(), "log/mnist_embed/vae/mnist/rep_dim=8/seed_$i") for i in 1:5],
    12 => [joinpath(AE_prefix(), "log/mnist_embed/vae/mnist/rep_dim=12/seed_$i") for i in 1:5]
);

using JLD

function parse_args(args)

    @assert iseven(length(args))

    class, dim, range, model = nothing, nothing, nothing, nothing

    for i in 1:2:length(args)
        key, value = args[i:i+1]
        if key == "--class" || key == "-c"
            class = try
                parse(Int, value)
            catch
                throw(ArgumentError("Class must be an integer between 0 and 9"))
            end
        elseif key == "--dim" || key == "-d"
            dim = try
                parse(Int, value)
            catch
                throw(ArgumentError("Dimension must be an integer"))
            end
        elseif key == "--range" || key == "-r"
            range = try
                parse(Int, value)
            catch
                throw(ArgumentError("range must be an integer"))
            end
        elseif key == "--model" || key == "-m"
            @assert value in ("ae", "vae")
            model = value
        else
            throw(ArgumentError(
            join([
                "Provide class ([0-9])",
                "dimension ([6,8,12])",
                "encoder model ([\"ae\", \"vae\"])",
                "upper range for number of hyperplanes ([50-160])"
                ],
                ", ", " and ")
            ))
        end
    end

    @show (class, dim, range, model)

    any(isnothing(arg) for arg in (class, dim, range, model)) && throw(ArgumentError("Provide class ([0-9]), dimension ([6,8,12]) and encoder model ([\"ae\", \"vae\"])."))

    return class, model, dim, 40:10:range
end

class, model, dim, hyperplanes = parse_args(ARGS)

if model == "ae"
    experiments = EXPERIMENTS_AE[dim]
elseif model == "vae"
    experiments = EXPERIMENTS_VAE[dim]
end

aucs = let experiments = experiments, class = class, hyperplanes = hyperplanes
    aucs = zeros(length(experiments), length(hyperplanes))
    for (seed, experiment) in enumerate(experiments)
        @info experiment
        (train_f, train_l) = read_data_AE(experiment, set="train")
        (test_f , test_l ) = read_data_AE(experiment, set="test" )

        for (i, hyper) in enumerate(hyperplanes)
            @info "class = $class, hyperplanes = $hyper"
            @time aucs[seed, i] = test_auc(train_f, train_l, test_f', test_l, class, hyperplanes=hyper, seed=seed)
        end
    end
    aucs
end

save("./log/aucs_$(model)_d=$(dim)_c=$(class).jld", "aucs", aucs, "hyperplanes", hyperplanes)
