for p in ("Knet","ArgParse","Images", "Distributions")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
# include(Pkg.dir("Knet","data","mnist.jl"))#, "imagenet.jl"))
include(Pkg.dir("Knet","data","imagenet.jl"))

module CelebA_Loader
using Knet
using Images
using ArgParse
using JLD2, FileIO


function main(args)
    o = parse_options(args)
    println("Data loading started")
    trndata = load_traindata(o[:trndata_dir]) # 202,599 instances i.e. shape: (32,32,3,202599)
    println("Data loading finished")
    println(summary(trndata))
    filename = "CelebA_traindata.jld2"
    filepath = joinpath(o[:outdir],filename)
    save(filepath, "xtrn", trndata)
    return trndata
end

function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "InfoGAN on CelebA."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=128; help="batch size")
        ("--trndata_dir"; default="/KUFS/scratch/ccengiz17/InfoGAN/datasets/CelebA/img_align_celeba"; help=("path for training dataset"))
        ("--zdim"; arg_type=Int; default=128; help="noise dimension")
        ("--cdim"; arg_type=Int; default=10; help=("Discrete code length."))
        ("--ndisc"; arg_type=Int; default=10; help=("number of discrete(categorical) codes"))
        ("--epochs"; arg_type=Int; default=100; help="# of training epochs")
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gridrows"; arg_type=Int; default=10)
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--optim"; default="Adam(;lr=0.0002, beta1=0.5)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; required=true; help="output dir for dataset matrix")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end

function imgload(imgdir, newsize)
    img = load(imgdir)
    img2 = imresize(img, newsize)
    img2 = channelview(img2)
    img2 = permutedims(img2, [2,3,1])
    img2 = Array{Float32}(img2);
end

function load_traindata(datadir; newsize=(32,32))
    x = []
    map(i->push!(x, imgload(joinpath(datadir,i), newsize)), readdir(datadir))
    #x = cat(4, x...)
    return x
end

function load_CelebA(atype,loadfile=nothing)
    @load loadfile xtrn
    return xtrn
end

splitdir(PROGRAM_FILE)[end] == "CelebA_loader.jl" && main(ARGS)

end # module
