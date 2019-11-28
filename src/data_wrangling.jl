using DelimitedFiles

function prepare_poisoned_data(data, data_labels; id=3, percentage_outliers=0.02)

#     target class
    target_idcs = findall(data_labels .== id);
    target_data = data[:, target_idcs];
    # @show size(target_data)

#     outlier class
    no_outs = ceil(Int, percentage_outliers * size(target_data,2))
    # @show no_outs

    outlier_idcs = Vector{Int}(undef, no_outs)
    for i in 1:no_outs
        k = 0
        label = id
        while label == id
            k = rand(1:length(data_labels))
            label = data_labels[k]
        end
        outlier_idcs[i] = k
    end

    outlier_data = data[:, outlier_idcs];
    outlier_labels = data_labels[outlier_idcs]

    prepared_labels = vcat(data_labels[target_idcs], data_labels[outlier_idcs])
    prepared_data = hcat(target_data, outlier_data)
    return prepared_data, prepared_labels
end

function labels01(labels, id)
    labels = deepcopy(labels)
    idcs = labels .== id
    labels[idcs] .= 0
    labels[.~idcs] .= 1
    return collect(labels)
end

AE_prefix() = joinpath(Base.dirname(Base.active_project()),
    "AE-PyTorch")

function read_data_AE(path; set="train")
    features = readdlm(joinpath(path, "embed_$set.csv"), ',', Float64)
    labels = Int.(readdlm(joinpath(path, "labels_$set.csv"), ',', Float64))
    return features, labels
end
