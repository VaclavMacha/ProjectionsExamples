module ProjectionsExamples

    using Projections, Distributions, LinearAlgebra, Statistics, Random, DataFrames, Plots, CSV

    export run_benchmarks, run_mu

    include("utilities.jl")
    include("examples.jl")

    include("run_find_mu.jl")
    include("run_simplex_mod1.jl")
    include("run_simplex_mod2.jl")
    include("run_philpott.jl")
    include("run_minimize_linear_on_simplex_l1.jl")
    include("run_minimize_linear_on_simplex_l2.jl")
    include("run_minimize_linear_on_simplex_lInf.jl")
    include("run.jl")

end # module
