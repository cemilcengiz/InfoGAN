for p in ("Knet","ArgParse","Images", "Distributions")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))
include(Pkg.dir("Knet","data","imagenet.jl"))

"""
    This example implements a Conditional INFOGAN on MNIST dataset.
"""
module ConditionalInfoGAN
using Knet
using Distributions
using Images
using ArgParse
using JLD2, FileIO

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load models, data, optimizers
    #cb = initcodebook(o[:atype], o[:cdim])
    wd, wg, wq, md, mg, mq = load_weights(o[:atype], o[:zdim], o[:cdim], o[:loadfile])
    wgq = Dict(:wg => wg, :wq => wq)
    xtrn,ytrn,xtst,ytst = Main.mnist() #60,000 instances
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; shuffle=true, xtype=o[:atype])  #length trnsize/mbsizae
    optd = map(wi->eval(parse(o[:optim])), wd)
    optgq = Dict(
                 :wg => map(wi->eval(parse(o[:optim])), wg),
                 :wq => map(wi->eval(parse(o[:optim])), wq)
                 )

    z = sample_noise(o[:atype],o[:zdim],prod(o[:gridsize]))
    #c = sample_c(o[:atype],o[:cdim],prod(o[:gridsize]))
    l0 = map(i->reshape(collect(1:10), 1, 10), 1:10)
    l1 = vec(vcat(l0...)) #00...011...122...23...99...9
    
    if o[:outdir] != nothing && !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"generations"))
    end

    # training
    println("training started..."); flush(STDOUT)
    for epoch = 1:o[:epochs]
        dlossval = glossval = qlossval = 0
        @time for (x,y) in dtrn
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            #c = sample_c(o[:atype],o[:cdim],length(y))
            #c = y_gold
            #train D
            dlossval += train_discriminator!(wd,wgq,md,mg,2x-1,y,noise,optd,o)
            #train G
            glossval += train_generator!(wgq,wd,mg,md,noise,y,optgq,o)
            #train Q
            qlossval += train_qnet!(wgq,mg,mq,noise,y,optgq,o)
        end
        dlossval /= length(dtrn); glossval /= length(dtrn); qlossval /= length(dtrn);
        println((:epoch,epoch,:dloss,dlossval,:gloss,glossval,:qloss,qlossval))
        flush(STDOUT)

        # save models and generations
        if o[:outdir] != nothing
            filename = @sprintf("%04d.png",epoch)
            filepath = joinpath(o[:outdir],"generations",filename)
            plot_generations(
                             wgq[:wg], mg, l1; z=z, savefile=filepath,
                             scale=o[:gridscale], gridsize=o[:gridsize])
            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
            save_weights(filepath,wd,wgq,md,mg,mq)
        end
    end

    return wd,wgq,md,mg,mq
end

function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "Conditional InfoGAN on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=100; help="batch size")
        ("--zdim"; arg_type=Int; default=62; help="noise dimension")
       ("--cdim"; arg_type=Int; default=10; help=("c length. Default is 10 for Mnist digits."))
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gridsize"; arg_type=Int; nargs=2; default=[10,10])
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--optim"; default="Adam(;lr=0.0002, beta1=0.5)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end

function load_weights(atype,zdim,cdim,loadfile=nothing)
    if loadfile == nothing
        wd, md = initwd(atype)
        wg, mg = initwg(atype,zdim)
        wq, mq = initwq(atype,cdim)
    else
        @load loadfile wd wg wq md mg mq
        wd = convert_weights(wd, atype)
        wg = convert_weights(wg, atype)
        wq = convert_weights(wq, atype)
        md = convert_moments(md, atype)
        mg = convert_moments(mg, atype)
        mq = convert_moments(mq, atype)
    end
    return wd, wg, wq, md, mg, mq
end

function save_weights(savefile,wd,wgq,md,mg,mq)
    save(savefile,
         "wd", convert_weights(wd),
         "wg", convert_weights(wgq[:wg]),
         "wq", convert_weights(wgq[:wq]),
         "md", convert_moments(md),
         "mg", convert_moments(mg),
         "mq", convert_moments(mq))
end

function convert_weights(w, atype=Array{Float32})
    w0 = map(wi->convert(atype, wi), w)
    w1 = convert(Array{Any}, w0)
end


function convert_moments(moments,atype=Array{Float32})
    clone = map(mi->bnmoments(), moments)
    for k = 1:length(clone)
        if moments[k].mean != nothing
            clone[k].mean = convert(atype, moments[k].mean)
        end

        if moments[k].var != nothing
            clone[k].var = convert(atype, moments[k].var)
        end
    end
    return convert(Array{Any,1}, clone)
end


function leaky_relu(x, alpha=0.1)
    pos = max(0,x)
    neg = min(0,x) * alpha
    return pos + neg
end

function sample_noise(atype,zdim,nsamples,mu=0.5,sigma=0.5)
    noise = convert(atype, randn(zdim,nsamples))
    #normalized = (noise-mu)/sigma
end

function sample_c(atype,cdim,nsamples,mu=0.5,sigma=0.5)
    d = Multinomial(1,10)  #number of outcomes is equal to the code length
    c = rand(d,nsamples)      #one-hot vectors.shape [nsamples,cdim]
    convert(atype, c)
end

function initwd(atype, winit=0.01)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(5,5,2,20))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    push!(w, winit*randn(5,5,20,50))
    push!(w, bnparams(50))
    push!(m, bnmoments())

    push!(w, winit*randn(500,800))
    push!(w, bnparams(500))
    push!(m, bnmoments())

    push!(w, winit*randn(1,500))
    push!(w, zeros(1,1))

    #include embed matrix
    push!(w, winit*randn(784,10))
    return convert_weights(w,atype), m
end

function dnet(w,x,y,m; training=true, alpha=0.1)
    a0 = w[end][:,y]
    a1 = vcat(a0, reshape(x, 784, size(x,4)))
    x0 = reshape(a1, 28, 28, 2, size(x,4))
    x1 = dlayer1(x0, w[1:2], m[1]; training=training)
    x2 = dlayer1(x1, w[3:4], m[2]; training=training)
    x3 = reshape(x2, 800,size(x2,4))
    x4 = dlayer2(x3, w[5:6], m[3]; training=training)
    x5 = w[end-2] * x4 .+ w[end-1]
    x6 = sigm.(x5)
end

function dlayer1(x0, w, m; stride=1, padding=0, alpha=0.1, training=true)
    x = conv4(w[1], x0; stride=stride, padding=padding)
    x = batchnorm(x, m, w[2]; training=training)
    x = leaky_relu.(x,alpha)
    x = pool(x; mode=2)
end

function dlayer2(x, w, m; training=true, alpha=0.1)
    x = w[1] * x
    x = batchnorm(x, m, w[2]; training=training)
    x = leaky_relu.(x, alpha)
end

function dloss(w,m,real_images,fake_images,ygold)
    yreal = dnet(w,real_images,ygold,m)
    real_loss = -log.(yreal+1e-8)
    yfake = dnet(w,fake_images,ygold,m)
    fake_loss = -log.(1-yfake+1e-8)
    return mean(real_loss+fake_loss)
end

dlossgradient = gradloss(dloss)
function train_discriminator!(wd,wgq,md,mg,real_images,ygold,noise,optd,o)
    fake_images = gnet(wgq[:wg],noise,ygold,mg; training=true)
    gradients, lossval = dlossgradient(wd,md,real_images,fake_images,ygold)
    update!(wd, gradients, optd)
    return lossval
end

function initwg(atype=Array{Float32}, zdim=62, embed=100, winit=0.01)
    w = Any[]
    m = Any[]

    # 2 dense layers combined with batch normalization layers
    push!(w, winit*randn(500,zdim+embed))
    push!(w, bnparams(500))
    push!(m, bnmoments())

    push!(w, winit*randn(800,500)) # reshape 4x4x16
    push!(w, bnparams(800))
    push!(m, bnmoments())

    # 3 deconv layers combined with batch normalization layers
    push!(w, winit*randn(2,2,50,50))
    push!(w, bnparams(50))
    push!(m, bnmoments())

    push!(w, winit*randn(5,5,20,50))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    push!(w, winit*randn(2,2,20,20))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    # final deconvolution layer
    push!(w, winit*randn(5,5,1,20))
    push!(w, winit*randn(1,1,1,1))

    #embedding layer for labels
    push!(w, winit*randn(embed,10))
    return convert_weights(w,atype), m
end

function gnet(wg,z,y,mg; training=true)  #add c
    x0 = vcat(z,wg[end][:,y])
    x1 = glayer1(x0, wg[1:2], mg[1]; training=training)
    x2 = glayer1(x1, wg[3:4], mg[2]; training=training)
    x3 = reshape(x2, 4,4,50,size(x2,2))
    x4 = glayer2(x3, wg[5:6], mg[3]; training=training)
    x5 = glayer3(x4, wg[7:8], mg[4]; training=training)
    x6 = glayer2(x5, wg[9:10], mg[5]; training=training)
    x7 = tanh.(deconv4(wg[end-2], x6) .+ wg[end-1])
end

function glayer1(x0, w, m; training=true)
    x = w[1] * x0
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function glayer2(x0, w, m; training=true)
    x = deconv4(w[1], x0; stride=2)
    x = batchnorm(x, m, w[2]; training=training)
end

function glayer3(x0, w, m; training=true)
    x = deconv4(w[1], x0)
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function gloss(wg,wd,mg,md,noise,labels)
    fake_images = gnet(wg,noise,labels,mg)
    ypred = dnet(wd,fake_images,labels,md)
    return -mean(log.(ypred+1e-8))
end

glossgradient = gradloss(gloss)

function train_generator!(wgq,wd,mg,md,noise,labels,optgq,o)
    gradients, lossval = glossgradient(wgq[:wg],wd,mg,md,noise,labels)
    update!(wgq[:wg],gradients,optgq[:wg])
    return lossval
end

function initwq(atype, cdim=10, winit=0.01)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(5,5,1,20))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    push!(w, winit*randn(5,5,20,50))
    push!(w, bnparams(50))
    push!(m, bnmoments())

    push!(w, winit*randn(500,800))
    push!(w, bnparams(500))
    push!(m, bnmoments())

    #FC.128-batchnorm-LRELU
    push!(w, winit*randn(128,500))
    push!(w, bnparams(128))
    push!(m, bnmoments())

    #FC.output
    push!(w, winit*randn(cdim,128))
    push!(w, zeros(cdim,1))
    return convert_weights(w,atype), m
end

function qnet(wq,x0,mq; training=true, alpha=0.1)
    x1 = dlayer1(x0, wq[1:2], mq[1]; training=training)
    x2 = dlayer1(x1, wq[3:4], mq[2]; training=training)
    x3 = reshape(x2, 800,size(x2,4))
    x4 = dlayer2(x3, wq[5:6], mq[3]; training=training)
    #FC.128-batchnorm-LRELU
    x5 = glayer1(x4, wq[7:8], mq[4]; training=training)
    #FC.output
    x6 = wq[end-1] * x5 .+ wq[end] #unnormalized scores
    x7 = exp.(logp(x6,1))  #softmax probs
end

function qloss(wgq,mg,mq,labels,noise)
    fake_images = gnet(wgq[:wg],noise,labels,mg; training=true)
    q_c_given_x = qnet(wgq[:wq],fake_images,mq) #log probs. shape 10xBatchsize
    cond_ent = nll(q_c_given_x, labels, 1)
   #cond_ent = -mean(sum(log.(q_c_given_x +1e-8) .* c, 1))
    return cond_ent
end

qlossgradient = gradloss(qloss)

function train_qnet!(wgq,mg,mq,noise,labels,optgq,o)
    gradients, lossval = qlossgradient(wgq,mg,mq,labels,noise)
    update!(wgq, gradients, optgq)
    return lossval
end

function plot_generations(
    wg, mg, labels; z=nothing, gridsize=(10,10), scale=1.0, savefile=nothing)
    if z == nothing
        nimg = prod(gridsize)
        zdim = size(wg[1],2)-100  #100 is embed length
        atype = wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
        z = sample_noise(atype,zdim,nimg)
    end
    output = Array(0.5*(1+gnet(wg,z,labels,mg; training=false)))
    images = map(i->output[:,:,:,i], 1:size(output,4))
    grid = Main.make_image_grid(images; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(Gray, grid))
    else
        save(savefile, grid)
    end
end

splitdir(PROGRAM_FILE)[end] == "cinfogan.jl" && main(ARGS)

end # module
