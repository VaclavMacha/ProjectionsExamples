function run_AATP1(l::AbstractArray{<:Real},
				   mtds::AbstractArray{<:Symbol} = [:secant, :bisection];
				   kwargs...)

	C1(n,m) = fill(2, length(n));
	C2(n,m) = fill(0.7, length(n));

	## precompile for small n, m
	n = [100]
	m = [200]
	run_AATP1(n, m, C1(n, m), C2(n, m), mtds; kwargs...)

	## normal run
	n = @. ceil(Int64, 0.3*10^l);
	m = @. ceil(Int64, 0.7*10^l);

	return n, m, run_AATP1(n, m, C1(n, m), C2(n, m), mtds; kwargs...)
end


function run_AATP1(ns::AbstractArray{<:Integer},
				   ms::AbstractArray{<:Integer},
				   C1s::AbstractArray{<:Real},
				   C2s::AbstractArray{<:Real},
				   mtds::AbstractArray{<:Symbol};
				   atol::Real = 1e-8,
				   kwargs...)

	map(mtds) do mtd
		f(n, m, C1, C2) = run_AATP1(n, m, C1, C2, mtd; atol = atol, kwargs...)
		return map(f, ns, ms, C1s, C2s) |> rows -> reduce(vcat, rows)
	end |> rows -> reduce(vcat, rows)
end


function run_AATP1(n::Integer,
				   m::Integer,
				   C1::Real,
				   C2::Real,
				   mtd::Symbol;
				   max_evals::Integer = 1,
				   p0_dist = Normal(1/2,1),
				   q0_dist = Normal(1/2,1),
				   r0_dist = Uniform(0,1),
			 	   seed::Integer = 1234,
				   kwargs...)

    Random.seed!(seed);

	function eval()
		p0 = rand(p0_dist, n)
		q0 = rand(q0_dist, m)
		r0 = rand(r0_dist)
		return Projections.solve_AATP1(p0, q0, r0, C1, C2; method = mtd, returnstats = true, kwargs...)
	end

	return run_benchmark(eval, :AATP1, mtd; max_evals = max_evals)
end
