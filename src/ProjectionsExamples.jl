module ProjectionsExamples

    using Projections, Distributions, LinearAlgebra, Statistics, Random, DataFrames, Plots, CSV

    export run_benchmarks, run_mu

    include("utilities.jl")
    include("examples.jl")

    include("run_find_mu.jl")
    include("run_AATP1.jl")
    include("run_AATP2.jl")
    include("run_philpott.jl")
    include("run_DRO_l1.jl")
    include("run_DRO_l2.jl")
    include("run_DRO_lInf.jl")
    include("run.jl")

end # module
