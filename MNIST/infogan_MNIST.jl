for p in ("Knet","ArgParse","Images", "Distributions")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl", "imagenet.jl"))

module InfoGAN_MNIST
using Knet
using Distributions
using Images
using ArgParse
using JLD2, FileIO

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load models, data, optimizers
    wd, wg, wq, md, mg, mq = load_weights(o[:atype], o[:zdim], o[:cdim], o[:gdim], o[:loadfile])
    wgq = Dict(:wg => wg, :wq => wq)
    xtrn,ytrn,xtst,ytst = Main.mnist() #60,000 instances
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; shuffle=true, xtype=o[:atype])  #length trnsize/batchsize
    optd = map(wi->eval(parse(o[:optim])), wd)
    optgq = Dict(
                 :wg => map(wi->Adam(;lr=0.003, beta1=0.5), wg),
                 :wq => map(wi->eval(parse(o[:optim])), wq)
                 )

    # For output grid
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
            z = sample_gauss(o[:atype],o[:zdim],length(y))
            c = sample_categoric(o[:atype],o[:cdim],length(y))
            g = sample_unif(o[:atype],o[:gdim],length(y))
            #train D
            dlossval += train_discriminator!(wd,wgq,md,mg,2x-1,z,c,g,optd,o)
            #train G
            glossval += train_generator!(wgq,wd,mg,md,z,c,g,optgq,o)
            #train Q
            qlossval += train_qnet!(wgq,mg,mq,z,c,g,optgq,o)
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
        "InfoGAN on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=64; help="batch size")
        ("--zdim"; arg_type=Int; default=62; help="noise dimension")
        ("--cdim"; arg_type=Int; default=10; help=("Discrete code length. Default is 10 for Mnist digits."))
        ("--gdim"; arg_type=Int; default=2; help=("Continuos code length. It models the mean and standard deviations of Gaussians."))
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=1; help="random seed")
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

function load_weights(atype,zdim,cdim,gdim,loadfile=nothing)
    if loadfile == nothing
        wd, md = initwd(atype)
        wg, mg = initwg(atype,zdim,cdim,gdim)
        wq, mq = initwq(atype,cdim,gdim)
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

function sample_gauss(atype,zdim,nsamples;mu=0,sigma=1)
    dist = Normal(mu,sigma)
    z = rand(dist, zdim, nsamples)
    convert(atype,z)
end

function sample_unif(atype,gdim,nsamples;l=-1,u=1)
    dist = Unif(l,u)
    g = rand(dist, gdim, nsamples)
    convert(atype, g)
end

function sample_categoric(atype,cdim,nsamples,mu=0.5,sigma=0.5)
    dist = Multinomial(1,cdim)   
    c = rand(dist,nsamples)   #one-hot vectors.shape [cdim,nsamples]
    convert(atype, c)
end

function initwd(atype, winit=0.01)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(4,4,1,64))
  
    push!(w, winit*randn(4,4,64,128))
    push!(w, bnparams(128))
    push!(m, bnmoments())
    
    push!(w, winit*randn(1024,3200))
    push!(w, bnparams(1024))
    push!(m, bnmoments())

    push!(w, winit*randn(1,1024))
    push!(w, zeros(1,1))
    return convert_weights(w,atype), m
end

function dnet(w,x,m; training=true, alpha=0.1)
    x1 = dlayer1(x, w[1])
    x2 = dlayer2(x1, w[2:3], m[1]; training=true)
    x2 = mat(x2) 
    x3 = dlayer3(x2, w[4:5], m[2]; training=true)
    x4 = w[end-1]*x3 .+ w[end]
    x5 = sigm.(x4) #??? 
end

function dlayer1(x0, w; stride=2, padding=0, alpha=0.1)
    x = conv4(w, x0; stride=stride, padding=padding)
    x = leaky_relu.(x,alpha)
end

function dlayer2(x0, w, m; stride=2, padding=0, alpha=0.1, training=true)
    x = conv4(w[1], x0; stride=stride, padding=padding)
    x = batchnorm(x, m, w[2]; training=training)
    x = leaky_relu.(x,alpha)
end

function dlayer3(x, w, m; training=true, alpha=0.1)
    x = w[1] * x
    x = batchnorm(x, m, w[2]; training=training)
    x = leaky_relu.(x, alpha)
end

function dloss(w,m,real_images,fake_images)
    yreal = dnet(w, real_images, m)
    real_loss = -log.(yreal+1e-8)
    yfake = dnet(w,fake_images,m)
    fake_loss = -log.(1-yfake+1e-8)
    return mean(real_loss+fake_loss)
end

dlossgradient = gradloss(dloss)


function train_discriminator!(wd,wgq,md,mg,real_images,z,c,g,optd,o)
    fake_images = gnet(wgq[:wg],mg,z,c,g; training=true)
    gradients, lossval = dlossgradient(wd,md,real_images,fake_images)
    update!(wd, gradients, optd)
    return lossval
end

function initwg(atype=Array{Float32}, zdim=62, cdim=10, gdim=2, winit=0.01)
    #input: z+c+g=62+10+2=74 dim
    w = Any[]
    m = Any[]

    # 2 dense layers combined with batch normalization layers
    push!(w, winit*randn(1024,zdim+cdim+gdim))
    push!(w, bnparams(1024))
    push!(m, bnmoments())
    
    push!(w, winit*randn(7*7*128, 1024))
    push!(w, bnparams(7*7*128))
    push!(m, bnmoments())
    
    # 1 deconv layer combined with batch normalization layer
    push!(w, winit*randn(4,4,64,128))
    push!(w, bnparams(64))
    push!(m, bnmoments())

    # final deconvolution layer
    push!(w, winit*randn(4,4,1,64))
    push!(w, winit*randn(1,1,1,1))
    return convert_weights(w,atype), m
end

function gnet(wg,mg,z,c,g; training=true)  #add c,g
    x0 = vcat(z,c,g)
    x1 = glayer1(x0, wg[1:2], mg[1]; training=training)
    x2 = glayer1(x1, wg[3:4], mg[2]; training=training)
    x3 = reshape(x2, 7,7,128,size(x2,2))
    x4 = glayer2(x3, wg[5:6], mg[3]; training=training)
    x5 = tanh.(deconv4(wg[end-1], x4) .+ wg[end])
end


function glayer1(x0, w, m; training=true)
    x = w[1] * x0
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function glayer2(x0, w, m; training=true)
    x = deconv4(w[1], x0; stride=2)
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function gloss(wg,wd,mg,md,z,c,g)
    fake_images = gnet(wg,mg,z,c,g)
    ypred = dnet(wd,fake_images,md)
    return -mean(log.(ypred+1e-8))
end

glossgradient = gradloss(gloss)

function train_generator!(wgq,wd,mg,md,z,c,g,optgq,o)
    gradients, lossval = glossgradient(wgq[:wg],wd,mg,md,z,c,g)
    update!(wgq[:wg],gradients,optgq[:wg])
    return lossval
end

function initwq(atype, cdim=10, gdim=2 ,winit=0.01)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(4,4,1,64))
  
    push!(w, winit*randn(4,4,64,128))
    push!(w, bnparams(128))
    push!(m, bnmoments())
    
    push!(w, winit*randn(1024,3200))
    push!(w, bnparams(1024))
    push!(m, bnmoments())

    #FC.128-batchnorm-LRELU
    push!(w, winit*randn(128,1024))
    push!(w, bnparams(128))
    push!(m, bnmoments())

    #FC output
    push!(w, winit*randn(cdim+gdim,128))
    push!(w, zeros(cdim+gdim,1))
    return convert_weights(w,atype), m
end

function qnet(wq,mq,x; training=true, alpha=0.1)
    cdim=10; gdim=2;
    x1 = dlayer1(x, wq[1])
    x2 = dlayer2(x1, wq[2:3], mq[1]; training=training)
    x2 = mat(x2) 
    x3 = dlayer3(x2, wq[4:5], mq[2]; training=training)
    #FC.128-batchnorm-LRELU
    x4 = glayer1(x3, wq[6:7], mq[3]; training=training)
    #FC.output
    x5 = wq[end-1]*x4 .+ wq[end] #shape: (cdim+gdim,batchsize)

    # Compute softmax for discrete code
    xc = x5[1:cdim,:];  xg = x5[1+cdim:end,:];
    xc = exp.(logp(xc,1))  #shape: (cdim,batchsize)

    # Make sure the continuous code learning the standard deviation is positive by computing its exponential
    xg[2,:] = exp.(xg[1,:]) #shape: (1,batchsize)
    #Dont touch the mean i.e. xg[1,:]
    return vcat(xc,xg)
end

function qloss(wgq,mg,mq,z,c,g)
    fake_images = gnet(wgq[:wg],mg,z,c,g; training=true)
    q_c_given_x = qnet(wgq[:wq],mq,fake_images) #shape (cdim+gdim,batchsize)
    cond_ent = -mean(sum(log.(q_c_given_x +1e-8) .* c, 1))
    return cond_ent
end

qlossgradient = gradloss(qloss)


function train_qnet!(wgq,mg,mq,z,c,g,optgq,o)
    gradients, lossval = qlossgradient(wgq,mg,mq,z,c,g)
    update!(wgq, gradients, optgq)
    return lossval
end

function plot_generations(
    wg, mg; z=nothing, c=nothing, g=nothing, gridsize=(10,10), scale=1.0, savefile=nothing)
    if z == nothing
        nimg = prod(gridsize)
        zdim = 62
        cdim = 10
        gdim = 2
        atype = wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
        z = sample_gauss(atype,zdim,nimg)
        c = sample_categoric(atype,cdim,nimg)
        # g is std,mean of a Gaussian
        #g = sample_gauss(atype,gdim,nimg)
    end
    
    output = Array(0.5*(1+gnet(wg,mg,z,c,g; training=false)))
    images = map(i->output[:,:,:,i], 1:size(output,4))
    grid = Main.make_image_grid(images; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(Gray, grid))
    else
        save(savefile, grid)
    end
end

splitdir(PROGRAM_FILE)[end] == "infogan_MNIST.jl" && main(ARGS)

end # module
