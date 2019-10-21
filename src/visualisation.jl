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

function reconstruct(verts; export_img=tempname()*".png", visualize_verts_scipt="../src/visualise_verts.py")
    if occursin(".", export_img)
        parts = split(export_img, ".")
        fn, ext = join(parts[1:end-1], "."), parts[end]
    else
        fn = export_img
    end

    csv_file = fn*".csv"
    @info "Writing points to" csv_file
    writedlm(csv_file, verts, ',')
    run(`python $visualize_verts_scipt $csv_file`)
    return csv_file*"_rec"
end
