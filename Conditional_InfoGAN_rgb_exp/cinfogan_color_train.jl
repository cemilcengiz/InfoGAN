include("cinfogan_color.jl")

include("data.jl")

wd, wgq, md, mg, mq = ConditionalInfoGAN.main("--epochs 1000 --seed 1 --outdir saved1");
