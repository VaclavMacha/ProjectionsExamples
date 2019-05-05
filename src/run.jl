function run_benchmarks(tablesavepath::String = "", plotsavepath::String = ""; kwargs...)


	l  = unique(vcat(3:0.2:4, 4:0.1:5));

	## AATP1, AATP2
	# l    = unique(vcat(3:0.2:4, 4:0.1:5, 5:0.05:6));
	mtds = [:secant, :bisection]

	n, m, tab1 = run_AATP1(l, mtds; kwargs...)
	n, m, tab2 = run_AATP2(l, mtds; kwargs...)
	tab_AATP12  = maketable(n+m, n, m, tab1, tab2; savename = joinpath(tablesavepath, "AATP1_AATP2.csv"))

	if !isempty(plotsavepath)
		for prb in [:AATP1, :AATP2]
			ttl1 = joinpath(plotsavepath, "times_$prb")
			ttl2 = joinpath(plotsavepath, "evals_$prb")
			ttl3 = joinpath(plotsavepath, "mu_$prb")
			experiments_plot(tab_AATP12, [:t],     [prb], mtds; title = ttl1, saveplot = true)
			experiments_plot(tab_AATP12, [:evals], [prb], mtds; title = ttl2, saveplot = true, xscale = :log10)
			experiments_plot(tab_AATP12, [:x],     [prb], mtds; title = ttl3, saveplot = true, xscale = :log10)
		end
	end


	## AATP2 with fixed C2
	# l    = unique(vcat(3:0.2:4, 4:0.1:5, 5:0.05:6));
	mtds = [:secant, :bisection]

	n, m, tab    = run_AATP2(l, mtds; fix = 100, kwargs...)
	tab_AATP2_fix = maketable(n+m, n, m, tab1, tab; savename = joinpath(tablesavepath, "AATP2_fix.csv"))

	if !isempty(plotsavepath)
		ttl1 = joinpath(plotsavepath, "times_AATP2_fix")
		ttl2 = joinpath(plotsavepath, "evals_AATP2_fix")
		ttl3 = joinpath(plotsavepath, "mu_AATP2_fix")
		experiments_plot(tab_AATP2_fix, [:t],     [:AATP2], mtds; title = ttl1, saveplot = true)
		experiments_plot(tab_AATP2_fix, [:evals], [:AATP2], mtds; title = ttl2, saveplot = true, xscale = :log10)
		experiments_plot(tab_AATP2_fix, [:x],     [:AATP2], mtds; title = ttl3, saveplot = true, xscale = :log10)
	end


	## DRO: l1, l2 and lInf norm
	# l    = unique(vcat(3:0.2:4, 4:0.1:5, 5:0.05:6));
	mtds = [:newton, :secant, :bisection]

	n, tab1    = run_DRO_lInf(l; kwargs...)
	n, tab2    = run_DRO_l1(l; kwargs...)
	n, tab3    = run_DRO_l2(l, mtds; kwargs...)
	tab_l12Inf = maketable(n, n, zero(n), tab1, tab2, tab3; savename = joinpath(tablesavepath, "DRO_l1_l2_lInf.csv"))

	if !isempty(plotsavepath)
		for prb in [:DROl1, :DROl2, :DROlInf]
			if prb ∈ [:DROl1, :DROlInf]
				mtds_in = [:none]
			else
				mtds_in = mtds
			end

			ttl1 = joinpath(plotsavepath, "times_$prb")
			ttl2 = joinpath(plotsavepath, "evals_$prb")
			ttl3 = joinpath(plotsavepath, "mu_$prb")
			experiments_plot(tab_l12Inf, [:t],     [prb], mtds_in; title = ttl1, saveplot = true)
			if prb == :DROl2
				experiments_plot(tab_l12Inf, [:evals], [prb], mtds_in; title = ttl2, saveplot = true, xscale = :log10)
				experiments_plot(tab_l12Inf, [:x],     [prb], mtds_in; title = ttl3, saveplot = true, xscale = :log10)
			end
		end
	end


l  = unique(vcat(2:0.1:3));
	## DRO l2 vs. Philpott
	# l    = unique(vcat(3:0.1:4, 4:0.05:5));
	mtds = [:newton, :secant, :bisection]

	n, tab1  = run_DRO_l2(l, mtds; kwargs...)
	n, tab2  = run_philpott(l; kwargs...)
	tab_phil = maketable(n, n, zero(n), tab1, tab2; savename = joinpath(tablesavepath, "DROl2_philpott.csv"))

	if !isempty(plotsavepath)
		ttl = joinpath(plotsavepath, "times_philpott")
		experiments_plot(tab_phil, [:t], [:philpott], [:none]; title = ttl, saveplot = true)
	end

	return tab_AATP12, tab_AATP2_fix, tab_l12Inf, tab_phil
end


function run_mu(tablesavepath::String = "", plotsavepath::String = "")
	l = 3;
	n = @. ceil(Int64, 0.3*10^l);
	m = @. ceil(Int64, 0.7*10^l);
	N = n + m

	C1 = 2;
	C2 = 0.7;
	C3 = ceil(Int64, m/10)
	ε  = 0.1

	x1 = range(2;   stop = 4,   length = 100)
	x2 = range(0.1; stop = 6,   length = 100)
	x3 = range(0.1; stop = 110, length = 100)

	tab1 = mu_AATP1(x1, n, m, C1, C2)
	tab2 = mu_AATP2(x2, n, m, C1, C3)
	tab3 = mu_DRO_l2(x3, N, ε)

	tab = hcat(tab1, tab2, tab3)

	if !isempty(tablesavepath)
		CSV.write(joinpath(tablesavepath, "find_mu.csv"), tab)
	end

	if !isempty(plotsavepath)
		for prb in [:AATP1, :AATP2, :DROl2]
			ttl = joinpath(plotsavepath, "find_mu_$prb")
			mu_plot(tab, [prb]; title = ttl, saveplot = true)
		end
	end

	return tab
end
