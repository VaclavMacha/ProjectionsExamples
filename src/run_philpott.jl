function run_philpott(l::AbstractArray{<:Real};
					  kwargs...)

	ε(n) = fill(0.1, length(n));

	## precompile for small n, m
	n = [100]
	run_philpott(n, ε(n); kwargs...)

	## normal run
	n = @. ceil(Int64, 10^l);

	return n, run_philpott(n, ε(n); kwargs...)
end


function run_philpott(ns::AbstractArray{<:Integer},
					  εs::AbstractArray{<:Real};
					  atol::Real = 1e-8,
					  kwargs...)

	map(ns, εs) do n, ε
		return run_philpott(n, ε; atol = atol, kwargs...)
	end |> rows -> reduce(vcat, rows)
end


function run_philpott(n::Integer,
					  ε::Real;
					  max_evals::Integer = 1,
					  p0_dist = Uniform(0,1),
					  c_dist = Normal(0,1),
					  seed::Integer = 1234,
					  atol::Real = 1e-8,
					  kwargs...)

    Random.seed!(seed);

	function eval()
		p0 = rand(p0_dist, n)
		p0 ./= sum(p0)
		c  = rand(c_dist, n)
		return Projections.philpott_optimized(p0, c, ε; atol = atol)
	end

	return run_benchmark(eval, :philpott; max_evals = max_evals)
end