function run_minimize_linear_on_simplex_l2(l::AbstractArray{<:Real},
										   mtds::AbstractArray{<:Symbol} = [:newton, :secant, :bisection];
										   kwargs...)

	ε(n) = fill(0.1, length(n));

	## precompile for small n, m
	n = [100]
	run_minimize_linear_on_simplex_l2(n, ε(n), mtds; kwargs...)

	## normal run
	n = @. ceil(Int64, 10^l);

	return n, run_minimize_linear_on_simplex_l2(n, ε(n), mtds; kwargs...)
end


function run_minimize_linear_on_simplex_l2(ns::AbstractArray{<:Integer},
										   εs::AbstractArray{<:Real},
										   mtds::AbstractArray{<:Symbol};
										   atol::Real = 1e-8,
										   kwargs...)
	map(mtds) do mtd
		map(ns, εs) do n, ε
			return run_minimize_linear_on_simplex_l2(n, ε, mtd; atol = atol, kwargs...)
		end |> rows -> reduce(vcat, rows)
	end |> rows -> reduce(vcat, rows)
end


function run_minimize_linear_on_simplex_l2(n::Integer,
										   ε::Real,
										   mtd::Symbol;
										   max_evals::Integer = 1,
										   p0_dist = Uniform(0,1),
										   c_dist = Normal(0,1),
										   seed::Integer = 1234,
										   kwargs...)

    Random.seed!(seed);

	function eval()
		p0 = rand(p0_dist, n)
		p0 ./= sum(p0)
		c  = rand(c_dist, n)
		c .-= maximum(c .+ 1)
		return Projections.minimize_linear_on_simplex_l2(p0, c, ε; method = mtd, returnstats = true, kwargs...)
	end

	return run_benchmark(eval, :l2, mtd; max_evals = max_evals)
end