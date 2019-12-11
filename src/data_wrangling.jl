using DelimitedFiles

function prepare_poisoned_data(data::AbstractMatrix, labels::AbstractVector, class; percentage_outliers=0.02)

#     target class
    class_idcs = findall(labels .== class)
    target_data = data[:, class_idcs]
    # @show size(target_data)

#     outlier class
    n_outs = ceil(Int, percentage_outliers * size(target_data, 2))
    # @show no_outs

    outlier_idcs = Vector{Int}(undef, n_outs)
    i = 1
    while i <= n_outs
        k = rand(1:length(labels))
        if labels[k] != class
            label = labels[k]
            outlier_idcs[i] = k
            i += 1
        end
    end

    outlier_data = data[:, outlier_idcs]
    outlier_labels = labels[outlier_idcs]

    prepared_labels = vcat(labels[class_idcs], labels[outlier_idcs])
    prepared_data = hcat(target_data, outlier_data)
    return prepared_data, prepared_labels
end

function labels01(labels, id)
    labels = deepcopy(labels)
    idcs = labels .== id
    labels[idcs] .= 0
    labels[.~idcs] .= 1
    return labels
end

AE_prefix() = joinpath(Base.dirname(Base.active_project()),
    "AE-PyTorch")

function read_data_AE(path; set="train")
    features = readdlm(joinpath(path, "embed_$set.csv"), ',', Float64)
    labels = Int.(readdlm(joinpath(path, "labels_$set.csv"), ',', Float64))
    return Matrix(transpose(features)), labels[:, 1]
end
