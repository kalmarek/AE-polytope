using Plots
using Plots.PlotMeasures
using Images

function imshow(m::AbstractMatrix{Float64})
    M = Gray.(m)
    return plot(M,
        xaxis=nothing, yaxis=nothing, legend=:none,
        yflip=true, transpose=true,
        aspect_ratio=1.0)
end

function visualize_digits(features; width=25, dpp=28, margin=0px)
    @assert size(features, 1) == 784
    k = size(features, 2)
    h, r = divrem(k, width)
    height = r == 0 ? h : h+1
    plts = [imshow(reshape(features[:,i], 28, 28)) for i in 1:k]
    zz = zeros(28,28)
    append!(plts, [imshow(zz) for _ in 1:(width*height - k)])
    plot(plts..., layout=grid(height, width), size=(width*dpp, height*dpp), margin=margin)
end

struct AE end
struct VAE end

function reconstruct(verts; export_file=tempname(), seed=1, vae=true)
    if endswith(".csv", export_file)
        export_file = export_file[1:end-4]
    end

    dim = size(verts, 2)
    input_file = export_file*".csv"
    output_file = export_file*"_rec.csv"
    @info "Writing points to" input_file
    writedlm(input_file, verts, ',')

    if vae
        reconstruct(VAE, input_file, output_file, dim, seed)
    else
        reconstruct( AE, input_file, output_file, dim, seed)
    end

    return output_file
end

function reconstruct(::Type{VAE}, input_file, output_file, dim, seed)
    run(`python ../AE-PyTorch/src/reconstruct.py $input_file $output_file mnist mnist_LeNet
    ../AE-PyTorch/log/mnist_embed/vae/mnist/rep_dim=$dim/seed_$seed
    --rep_dim $dim --device cpu --seed $seed --ae_model_type vae`)
end

function reconstruct(::Type{ AE}, input_file, output_file, dim, seed)
    run(`python ../AE-PyTorch/src/reconstruct.py $input_file $output_file mnist mnist_LeNet
    ../AE-PyTorch/log/mnist_embed/autoencoder/mnist/rep_dim=$dim/seed_$seed
    --rep_dim $dim --device cpu --seed $seed --ae_model_type vanilla_ae`)
end
