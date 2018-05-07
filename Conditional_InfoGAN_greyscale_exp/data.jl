module Shapes

using Luxor
using Colors
using ColorSchemes
using ArgParse
using ProgressMeter
using JSON

include(Pkg.dir("Knet","data","imagenet.jl"))

let
    global COLORS
    COLORS = Dict(
                  "black" => ("black",),
                  "blue" => ("blue1", "blue2", "blue3", "blue4"),
                  "cyan" => ("cyan1", "cyan2", "cyan3", "cyan4"),
                  "gray" => ("gray30", "gray40", "gray50", "gray60", "gray70"),
                  "green" => ("green1", "green2", "green3", "green4", "green"),
                  "orange" => ("darkorange", "darkorange1", "darkorange2",
                               "darkorange3", "darkorange4"),
    "pink" => ("deeppink1", "deeppink2", "deeppink3", "deeppink4"),
    "purple" => ("purple1", "purple2", "purple3", "purple4"),
    "red" => ("red1", "red2", "red3", "red4"),
    "white" => ("white",),
    "yellow" => ("yellow1", "gold1")
    )

    # get actual RGB values - essential for random color sampling
    for (color,shades) in COLORS
        shades = map(s->Colors.color_names[s], shades)
        shades = map(s->map(c->c/255, s), shades)
        shades = map(s->RGB(s...), shades)
        shades = collect(shades)
        shades = sortcolorscheme(shades)
        COLORS[color] = shades
    end

    # add brown additionally - peh
    brown = [(123,63,0),(107,68,35),(128,70,27),(150,75,0),(139,69,19)]
    brown = map(b->map(x->x/255, b), brown)
    brown = sortcolorscheme(collect(map(b->RGB(b...), brown)))
    COLORS["brown"] = brown
end


POLYGONS = (
            (:triangle, 3, 11),
            (:pentagon, 5, 11),
            (:hexagon, 6, 0),
            (:heptagon, 7, 11),
            (:octagon, 8, 0)
            )

for (F, N, O) in POLYGONS
    @eval begin
        function $F(x, y, radius, action=:fill, orientation=$O, o...)
            ngon(x, y, radius, $N, orientation, action, o...)
        end

        function $F(p::Point, radius, action=:fill, orientation=$O, o...)
            ngon(p, radius, $N, orientation, action, o...)
        end
    end
end


function square(x, y, radius, action=:fill, o...)
    box(x, y, 2*radius, 2*radius, action, o...)
end


function square(p::Point, radius, action=:fill, o...)
    box(p, 2*radius, 2*radius, action, o...)
end


function _star(x, y, radius, action=:fill)
    star(x, y, radius, 5, 0.5, 1, action)
end


function _star(p::Point, radius, action=:fill)
    star(p, radius, 5, 0.5, 1, action)
end


SHAPES = Dict(
              :triangle => triangle,
              :square => square,
              :pentagon => pentagon,
              :hexagon => hexagon,
              :heptagon => heptagon,
              :octagon => octagon,
              :circle => circle,
              :star => _star
              )


# as radius lengths, edge = 2radius
SIZES = Dict(
             :small => 32,
             :medium => 64,
             :large => 96
             )


SIZE_SYNONYMS = Dict(
                     :small => ("small", "tiny", "little"),
                     :medium => ("medium", "moderate", "middle-sized"),
                     :large => ("large", "big", "huge")
                     )


WIDENESS_DICT = Dict(
                     :max_ratio => 2.4,
                     :min_ratio => 1.2
                     )


function clip(x0, centroid, threshold)
    x1 = min(x0, centroid+threshold)
    x2 = max(x1, centroid-threshold)
end


function create_single_shape(W, H, shape, color, shape_size, bgcolor,
                             filepath, size_dict=SIZES, buffers=(2.,1.))
    b1, b2 = buffers

    # sample radius
    size_sigma = div(size_dict[:large] - size_dict[:medium], 2b1+b2)
    radius_mean = size_dict[shape_size]
    radius = size_dict[shape_size] + size_sigma * randn()
    radius = clip(radius, radius_mean, b1*size_sigma)

    # sample location
    location_sigma1 = div(W-2radius,2b1+b2)
    location_sigma2 = div(H-2radius,2b1+b2)
    x = clip(location_sigma1*randn(), 0, b1*location_sigma1)
    y = clip(location_sigma2*randn(), 0, b1*location_sigma2)

    # sample color
    shade = get(COLORS[color], rand())

    # create shape
    Drawing(W,H,abspath(filepath))
    background(bgcolor)
    origin()
    setcolor(shade)
    SHAPES[shape](x,y,radius,:fill)
    finish()
end


function create_phrase(shape, color, sh_size)
    phrase = Array{String}([])

    # shape only
    push!(phrase, "a" *  " $shape")

    # shape and one additional attribute
    map(sz->push!(phrase, "a" * " $sz" * " $shape"), SIZE_SYNONYMS[sh_size])
    push!(phrase, "a" * " $color" * " $shape")

    # shape and two additional attributes
    map(sz->push!(phrase, "a" * " $sz" * " $color" * " $shape"),
        SIZE_SYNONYMS[sh_size])
    map(sz->push!(phrase, "a" * " $color" * " $sz" * " $shape"),
        SIZE_SYNONYMS[sh_size])

    return phrase
end


function create_entry(shape, color, shape_size, shape_id)
    Dict(
         "shape"    => string(shape),
         "color"    => color,
         "size"     => string(shape_size),
         "id"       => shape_id,
         "filename" => shape_id*".png",
         "phrase"   => create_phrase(shape, color, shape_size)
         )
end


function process_image(img, imgsize)
    y0 = Images.imresize(img, imgsize)
    y1 = permutedims(channelview(y0), (2,1))
    #y1 = permutedims(channelview(y0), (3,2,1))
    y2 = convert(Array{Float32}, y1)
    #y3 = reshape(y2[:,:,1:3], (imgsize...,3,1))
    y3 = reshape(y2, (imgsize...,1,1))
    y4 = permutedims(y3, (2,1,3,4))
end


function load_images(datadir, filenames, imgsize=(64,64))
    images = map(f->load_image(datadir, f, imgsize), filenames)
end


function load_image(datadir, filename, imgsize=(64,64))
    imgdir = joinpath(abspath(datadir), "images")
    raw_image = load(joinpath(imgdir, filename))
    #processed_image = process_image(raw_image, imgsize)
    processed_image = process_image(Gray.(raw_image), imgsize)
end


function load_data(datadir)
    datafile = joinpath(abspath(datadir), "dataset.json")
    JSON.parsefile(datafile)
end


function main(args)
    s = ArgParseSettings()
    s.description = "Generate SHAPES dataset."

    @add_arg_table s begin
        ("--savedir"; required=true; help="where to generated dataset")
        ("--nsamples"; default=100; arg_type=Int;
         help="number of instances for a specific shape")
        ("--imgsize"; nargs=2; default=[256,256]; arg_type=Int;
         help="window size in pixels")
        ("--radiuslength"; nargs=3; default=[32,64,92];
         help="radius length in pixels")
        ("--bgcolor"; default="black"; help="background color")
        ("--seed"; default=-1; arg_type=Int; help="random seed")
        ("--idlen"; default=10; arg_type=Int)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:savedir] = abspath(o[:savedir])
    o[:seed] > 0 && srand(o[:seed])
    W, H = o[:imgsize]

    function create_shape(shape, color, shape_size, file)
        create_single_shape(W, H, shape, color, shape_size, o[:bgcolor], file)
    end

    imgdir = joinpath(o[:savedir], "images")
    if !isdir(o[:savedir])
        mkpath(o[:savedir])
        mkpath(imgdir)
    else
        println("Save directory exists. Remove it first.")
        return
    end

    instances = []
    #(shapes, colors, sizes) = map(
    #                              keys, (SHAPES, filter((k,v)->k!=o[:bgcolor], COLORS), SIZES))
    shapes = [:triangle,:square,:circle,:star]
    colors = ["white"]
    sizes = [:large]

    p = Progress(mapreduce(length, *, (shapes, colors, sizes)), 1)
    for (shape, color, shape_size) in Base.product(shapes,colors,sizes)
        for k = 1:o[:nsamples]
            shape_id = randstring(o[:idlen])
            filename = shape_id*".png"
            filepath = joinpath(imgdir, filename)
            while isfile(filepath)
                shape_id = randstring(o[:idlen])
                filename = shape_id*".png"
                filepath = joinpath(imgdir, filename)
            end

            create_shape(shape, color, shape_size, filepath)
            entry = create_entry(shape, color, shape_size, shape_id)
            push!(instances, entry)
        end
        next!(p)
    end
    
    dataset = Dict(
                   "title"      => "shapes",
                   "datetime"   => string(now()),
                   "nsamples"   => o[:nsamples],
                   "imgsize"    => o[:imgsize],
                   "seed"       => o[:seed],
                   #"shapes"     => map(string, keys(SHAPES)),
                   #"colors"     => COLORS,
                   #"radius"     => SIZES,
                   "shapes"     => map(string, shapes),
                   "colors"     => colors,
                   "radius"     => sizes,
                   "ninstances" => length(instances),
                   "instances"  => instances
    )

    open(joinpath(o[:savedir], "dataset.json"), "w") do f
        write(f, json(dataset))
    end
end

splitdir(PROGRAM_FILE)[end] == "data.jl" && main(ARGS)

end
