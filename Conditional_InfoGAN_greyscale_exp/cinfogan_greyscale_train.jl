include("cinfogan_greyscale.jl")

include("data.jl")

wd, wgq, md, mg, mq = ConditionalInfoGAN.main("--epochs 300 --seed 1 --outdir saved1");
