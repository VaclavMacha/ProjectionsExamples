function run_DRO_l1(l::AbstractArray{<:Real}; kwargs...)

	ε(n) = fill(0.1, length(n));

	## precompile for small n, m
	n = [100]
	run_DRO_l1(n, ε(n); kwargs...)

	## normal run
	n = @. ceil(Int64, 10^l);

	return n, run_DRO_l1(n, ε(n); kwargs...)
end


function run_DRO_l1(ns::AbstractArray{<:Integer}, εs::AbstractArray{<:Real}; kwargs...)

	map(ns, εs) do n, ε
		return run_DRO_l1(n, ε; kwargs...)
	end |> rows -> reduce(vcat, rows)
end


function run_DRO_l1(n::Integer,
				    ε::Real;
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
		return Projections.solve_DRO_l1(p0, c, ε)
	end

	return run_benchmark(eval, :DROl1; max_evals = max_evals)
end
