for p in ("Knet","ArgParse","Images", "Distributions")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
# include(Pkg.dir("Knet","data","mnist.jl"))#, "imagenet.jl"))
include(Pkg.dir("Knet","data","imagenet.jl"))

module InfoGAN_CelebA
using Knet
using Distributions
using MAT
using Images
using ArgParse
using JLD2, FileIO
using MLDatasets

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    #load training data
    @load o[:trndata_dir] xtrn 
    
    # load models, data, optimizers
    wd, wg, wq, md, mg, mq = load_weights(o[:atype], o[:zdim], o[:cdim], o[:ndisc], o[:loadfile])
    wgq = Dict(:wg => wg, :wq => wq)
    #length trnsize/batchsize
    optd = map(wi->eval(parse(o[:optim])), wd)
    optgq = Dict(
                 :wg => map(wi->Adam(;lr=0.003, beta1=0.5), wg),
                 :wq => map(wi->eval(parse(o[:optim])), wq)
                 )
    
    disc_embed_mat = init_disc_embed(o[:atype], o[:cdim])
    ##
    ny = 10 
    # Fix some noise and code to check the GAN output
    z_fix = sample_gauss(o[:atype],o[:zdim],ny*o[:gridrows])

    # Discrete codes for example generations
    l0 = map(i->reshape(collect(1:ny), 1, ny), 1:o[:gridrows]) 
    l1 = vec(vcat(l0...)) #[11...122...23...99...9910...10] vector with lenght ny*o[:gridsize]
    cl_rand = sample_categoric(o[:atype],o[:cdim],ny*o[:gridrows])
    cl = convert(o[:atype], disc_embed_mat[:,l1])
    c_fix = cl
    for i=1:9
        c_fix = vcat(c_fix, cl_rand) # control one disc r.v. the others are varying
    end 
    
    if o[:outdir] != nothing && !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"generations"))
    end

    # training
    println("training started..."); flush(STDOUT)
    L = length(xtrn)
    for epoch = 1:o[:epochs]
        dlossval = glossval = qlossval = 0
        indices = shuffle!(Array(1:L))
        @time for k=1:o[:batchsize]:L
            #make minibatch
            lo, up = k, min(L, k+o[:batchsize]-1)
            ind = indices[lo:up]
            batch_images = cat(4, xtrn[ind]...)
            x = convert(o[:atype], batch_images)
            z = sample_gauss(o[:atype],o[:zdim],size(x,4))
            c = sample_categoric_multiple(o[:atype],o[:cdim],size(x,4);n=10)
            #train D
            dlossval += train_discriminator!(wd,wgq,md,mg,2x-1,z,c,optd,o)
            #train G
            glossval += train_generator!(wgq,wd,mg,md,z,c,optgq,o)
            #train Q
            qlossval += train_qnet!(wgq,mg,mq,z,c,optgq,o)
        end
        dlossval /= L; glossval /= L; qlossval /= L;
        println((:epoch,epoch,:dloss,dlossval,:gloss,glossval,:qloss,qlossval))
        flush(STDOUT)

        # save models and generations
        if o[:outdir] != nothing #&& epoch % 5==0
            filename = @sprintf("%04d.png",epoch)
            filepath = joinpath(o[:outdir],"generations",filename)
            generate_and_plot(
                              wgq[:wg], mg; savefile=filepath, z=z_fix, c=c_fix,
                              scale=o[:gridscale], gridsize=(ny, o[:gridrows]))
            
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
        "InfoGAN on CelebA."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=128; help="batch size")
        ("--trndata_dir"; default="/KUFS/scratch/ccengiz17/InfoGAN/datasets/CelebA_jld2/CelebA_traindata.jld2"; help=("path for training dataset"))
        ("--zdim"; arg_type=Int; default=128; help="noise dimension")
        ("--cdim"; arg_type=Int; default=10; help=("Discrete code length."))
        ("--ndisc"; arg_type=Int; default=10; help=("number of discrete(categorical) codes"))
        ("--epochs"; arg_type=Int; default=100; help="# of training epochs")
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gridrows"; arg_type=Int; default=10)
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

function imgload(imgdir, newsize)
    img = load(imgdir)
    img2 = imresize(img, newsize)
    img2 = channelview(img2)
    img2 = permutedims(img2, [2,3,1])
    img2 = Array{Float32}(img2);
end

function load_traindata(datadir, batch_names; new_size=(32,32))
    x = []
    map(i->push!(x, imgload(joinpath(datadir,i), new_size)), batch_names)
    x = cat(4, x...)
    return x
end
    
function load_weights(atype,zdim,cdim,ndisc,loadfile=nothing)
    if loadfile == nothing
        wd, md = initwd(atype)
        wg, mg = initwg(atype,zdim,cdim,ndisc)
        wq, mq = initwq(atype,cdim,ndisc)
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

function sample_unif(atype,gdim,nsamples;l=-1,u=1)
    dist = Uniform(l,u)
    g = rand(dist, gdim, nsamples)
    convert(atype, g)
end

sample_gauss = sample_unif

function sample_categoric(atype,cdim,nsamples,mu=0.5,sigma=0.5)
    dist = Multinomial(1,cdim)   
    c = rand(dist,nsamples)   #one-hot vectors.shape [cdim,nsamples]
    convert(atype, c)
end

function sample_categoric_multiple(atype,cdim,nsamples,mu=0.5,sigma=0.5;n=4)
    c = []
    map(x->push!(c,sample_categoric(atype,cdim,nsamples,mu,sigma)), 1:n)
    vcat(c...)
end

function init_disc_embed(atype, code_len)
    cb = atype(eye(code_len))
end  


function initwd(atype, winit=0.02)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(4,4,3,64))
    
    push!(w, winit*randn(4,4,64,128))
    push!(w, bnparams(128))
    push!(m, bnmoments())

    push!(w, winit*randn(4,4,128,256))
    push!(w, bnparams(256))
    push!(m, bnmoments())

    push!(w, winit*randn(1,1024))
    push!(w, zeros(1,1))
    return convert_weights(w,atype), m
end

function dnet(w,x,m; training=true, alpha=0.1)
    x1 = dlayer1(x, w[1])
    x2 = dlayer2(x1, w[2:3], m[1]; training=true)
    x3 = dlayer2(x2, w[4:5], m[2]; training=true)
    x3 = mat(x3) # (1024,Batchize)
    x4 = w[end-1]*x3 .+ w[end]
    x5 = sigm.(x4)
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


function train_discriminator!(wd,wgq,md,mg,real_images,z,c,optd,o)
    fake_images = gnet(wgq[:wg],mg,z,c; training=true)
    gradients, lossval = dlossgradient(wd,md,real_images,fake_images)
    update!(wd, gradients, optd)
    return lossval
end

function initwg(atype=Array{Float32}, zdim=128, cdim=10, ndisc=10, winit=0.02)
    #input: z+c*ndisc=128+10*10=228 dim
    w = Any[]
    m = Any[]
    #fc-relu-bn
    push!(w, winit*randn(2*2*448, zdim+cdim*ndisc))
    push!(w, bnparams(2*2*448))
    push!(m, bnmoments())
    
    #upconv-relu-bn
    push!(w, winit*randn(4,4,256,448))
    push!(w, bnparams(256))
    push!(m, bnmoments())

    #upconv-relu-bn
    push!(w, winit*randn(4,4,128,256))

    #upconv-relu-bn
    push!(w, winit*randn(4,4,64,128))

    #upconv-tanh
    push!(w, winit*randn(4,4,3,64))
    push!(w, zeros(1,1,3,1))
    return convert_weights(w,atype), m
end

function gnet(wg,mg,z,c; training=true)  #add c
    x0 = vcat(z,c)
    x1 = glayer1(x0, wg[1:2], mg[1]; training=training) #(1792,bs)
    x2 = reshape(x1, 2,2,448,size(x1,2)) #(2,2,448,bs)
    x3 = glayer2(x2, wg[3:4], mg[2]; padding=1, training=training) #(4,4,256,bs)
    x4 = glayer3(x3, wg[5]; padding=1) #(8,8,128,bs)
    x5 = glayer3(x4, wg[6]; padding=1) #(16,16,64,bs)
    x6 = tanh.(deconv4(wg[end-1], x5; stride=2, padding=1) .+ wg[end]) #(32,32,3,bs)
    return x6
end

 
function glayer1(x0, w, m; training=true)
    x = w[1] * x0
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function glayer2(x0, w, m; padding=1,stride=2, training=true)
    x = deconv4(w[1], x0; padding=padding, stride=stride)
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function glayer3(x0, w; padding=1,stride=2)
    x = deconv4(w, x0; padding=padding, stride=stride)
    x = relu.(x)
end

function gloss(wg,wd,mg,md,z,c)
    fake_images = gnet(wg,mg,z,c)
    ypred = dnet(wd,fake_images,md)
    return -mean(log.(ypred+1e-8))
end

glossgradient = gradloss(gloss)

function train_generator!(wgq,wd,mg,md,z,c,optgq,o)
    gradients, lossval = glossgradient(wgq[:wg],wd,mg,md,z,c)
    update!(wgq[:wg],gradients,optgq[:wg])
    return lossval
end

function initwq(atype, cdim=10, ndisc=10, winit=0.02)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(4,4,3,64))
    
    push!(w, winit*randn(4,4,64,128))
    push!(w, bnparams(128))
    push!(m, bnmoments())

    push!(w, winit*randn(4,4,128,256))
    push!(w, bnparams(256))
    push!(m, bnmoments())

    #FC.128-batchnorm-LRELU
    push!(w, winit*randn(128,1024))
    push!(w, bnparams(128))
    push!(m, bnmoments())

    #FC output
    push!(w, winit*randn(ndisc*cdim, 128))
    push!(w, zeros(ndisc*cdim, 1))
    return convert_weights(w,atype), m

end

function qnet(w,m,x; training=true, alpha=0.1)
    cdim=10; ndisc=10;
    x1 = dlayer1(x, w[1])
    x2 = dlayer2(x1, w[2:3], m[1]; training=true)
    x3 = dlayer2(x2, w[4:5], m[2]; training=true)
    x3 = mat(x3) # (1024,Batchize)
    #FC.128-batchnorm-LRELU
    x4 = glayer1(x3, w[6:7], m[3]; training=training) # (128,bs)
    #FC.output
    x5 = w[end-1]*x4 .+ w[end] #shape: (cdim*ndisc,bs)
    
    # Apply softmax for discrete code
    xc = exp.(logp(x5,1))  #shape: (cdim,batchsize)

    return xc
end

function qloss(wgq, mg, mq, z, c; atype=KnetArray{Float32}, fix_std=true)
    fake_images = gnet(wgq[:wg],mg,z,c; training=true)
    q_c_given_x_disc = qnet(wgq[:wq],mq,fake_images)

    #discrete code
    cdim=10;     ndisc = 10;  
    disc_loss = 0;
    for i=1:ndisc
        cc = c[(i-1)*cdim+1 : i*cdim, :]
        disc_cross_ent = -mean(sum(log.(q_c_given_x_disc[(i-1)*cdim+1 : i*cdim, :] +1e-8) .* cc, 1))
        disc_ent = -mean(sum(log.(0.1+1e-8) .* cc, 1))
        disc_loss += -disc_ent + disc_cross_ent  # negative of mutual info
    end
    
    return disc_loss
end

qlossgradient = gradloss(qloss)


function train_qnet!(wgq,mg,mq,z,c,optgq,o)
    gradients, lossval = qlossgradient(wgq,mg,mq,z,c)
    update!(wgq, gradients, optgq)
    return lossval
end

function generate_images(wg, mg, z=nothing, c=nothing)
    nimg = size(wg[1],2)
    atype = wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
    if z == nothing 
        zdim = 128
        z = sample_gauss(atype,zdim,nimg)
    end
    
    if c == nothing
        cdim = 10; ndisc=10;
        c = sample_categoric_multiple(atype,cdim,nimg;n=10)
    end
    output = Array(0.5*(1+gnet(wg,mg,z,c; training=false)))
    images = map(i->output[:,:,:,i], 1:size(output,4))
end

function plot_images(images, gridsize=(8,8), scale=1.0, savefile=nothing)
    grid = Main.make_image_grid(images; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(RGB, grid))
        #display(colorview(Gray, grid))
    else
        save(savefile, colorview(RGB, grid))
        #save(savefile, colorview(Gray, grid))
    end
end


function generate_and_plot(wg, mg; z=nothing, c=nothing, savefile=nothing, gridsize=(8,8), scale=1.)
    images = generate_images(wg, mg, z, c)
    plot_images(images, gridsize, scale, savefile)
end

splitdir(PROGRAM_FILE)[end] == "infogan_CelebA.jl" && main(ARGS)

end # module
