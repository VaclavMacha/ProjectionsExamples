function run_benchmarks(tablesavepath::String = "", plotsavepath::String = ""; kwargs...)


	l  = unique(vcat(3:0.2:4, 4:0.1:5));

	## Simplex mod1 vs. mod2
	# l    = unique(vcat(3:0.2:4, 4:0.1:5, 5:0.05:6));
	mtds = [:secant, :bisection]

	n, m, tab1 = run_simplex_mod1(l, mtds; kwargs...)
	n, m, tab2 = run_simplex_mod2(l, mtds; kwargs...)
	tab_mod12  = maketable(n+m, n, m, tab1, tab2; savename = joinpath(tablesavepath, "mod1_vs_mod2.csv"))

	if !isempty(plotsavepath)
		for prb in [:mod1, :mod2]
			ttl1 = joinpath(plotsavepath, "times_$prb")
			ttl2 = joinpath(plotsavepath, "evals_$prb")
			ttl3 = joinpath(plotsavepath, "mu_$prb")
			experiments_plot(tab_mod12, [:t],     [prb], mtds; title = ttl1, saveplot = true)
			experiments_plot(tab_mod12, [:evals], [prb], mtds; title = ttl2, saveplot = true, xscale = :log10)
			experiments_plot(tab_mod12, [:x],     [prb], mtds; title = ttl3, saveplot = true, xscale = :log10)
		end
	end


	## Simplex mod1 vs. mod2
	# l    = unique(vcat(3:0.2:4, 4:0.1:5, 5:0.05:6));
	mtds = [:secant, :bisection]

	n, m, tab    = run_simplex_mod2(l, mtds; fix = 100, kwargs...)
	tab_mod2_fix = maketable(n+m, n, m, tab1, tab; savename = joinpath(tablesavepath, "mod2_fix.csv"))

	if !isempty(plotsavepath)
		ttl1 = joinpath(plotsavepath, "times_mod2_fix")
		ttl2 = joinpath(plotsavepath, "evals_mod2_fix")
		ttl3 = joinpath(plotsavepath, "mu_mod2_fix")
		experiments_plot(tab_mod2_fix, [:t],     [:mod2], mtds; title = ttl1, saveplot = true)
		experiments_plot(tab_mod2_fix, [:evals], [:mod2], mtds; title = ttl2, saveplot = true, xscale = :log10)
		experiments_plot(tab_mod2_fix, [:x],     [:mod2], mtds; title = ttl3, saveplot = true, xscale = :log10)
	end


	## Minimize linear on simplex for l1, l2 and lInf norm
	# l    = unique(vcat(3:0.2:4, 4:0.1:5, 5:0.05:6));
	mtds = [:newton, :secant, :bisection]

	n, tab1    = run_minimize_linear_on_simplex_lInf(l; kwargs...)
	n, tab2    = run_minimize_linear_on_simplex_l1(l; kwargs...)
	n, tab3    = run_minimize_linear_on_simplex_l2(l, mtds; kwargs...)
	tab_l12Inf = maketable(n, n, zero(n), tab1, tab2, tab3; savename = joinpath(tablesavepath, "l1_vs_l2_vs_lInf.csv"))

	if !isempty(plotsavepath)
		for prb in [:l1, :l2, :lInf]
			if prb ∈ [:l1, :lInf]
				mtds_in = [:none]
			else
				mtds_in = mtds
			end

			ttl1 = joinpath(plotsavepath, "times_$prb")
			ttl2 = joinpath(plotsavepath, "evals_$prb")
			ttl3 = joinpath(plotsavepath, "mu_$prb")
			experiments_plot(tab_l12Inf, [:t],     [prb], mtds_in; title = ttl1, saveplot = true)
			if prb == :l2
				experiments_plot(tab_l12Inf, [:evals], [prb], mtds_in; title = ttl2, saveplot = true, xscale = :log10)
				experiments_plot(tab_l12Inf, [:x],     [prb], mtds_in; title = ttl3, saveplot = true, xscale = :log10)
			end
		end
	end


l  = unique(vcat(2:0.1:3));
	## Minimize linear on simplex for l2 and philpott
	# l    = unique(vcat(3:0.1:4, 4:0.05:5));
	mtds = [:newton, :secant, :bisection]

	n, tab1  = run_minimize_linear_on_simplex_l2(l, mtds; kwargs...)
	n, tab2  = run_philpott(l; kwargs...)
	tab_phil = maketable(n, n, zero(n), tab1, tab2; savename = joinpath(tablesavepath, "l2_vs_philpott.csv"))

	if !isempty(plotsavepath)
		ttl = joinpath(plotsavepath, "times_philpott")
		experiments_plot(tab_phil, [:t], [:philpott], [:none]; title = ttl, saveplot = true)
	end

	return tab_mod12, tab_mod2_fix, tab_l12Inf, tab_phil
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

	tab1 = mu_simplex_mod1(x1, n, m, C1, C2)
	tab2 = mu_simplex_mod2(x2, n, m, C1, C3)
	tab3 = mu_l2(x3, N, ε)

	tab = hcat(tab1, tab2, tab3)

	if !isempty(tablesavepath)
		CSV.write(joinpath(tablesavepath, "find_mu.csv"), tab)
	end

	if !isempty(plotsavepath)
		for prb in [:mod1, :mod2, :l2]
			ttl = joinpath(plotsavepath, "find_mu_$prb")
			mu_plot(tab, [prb]; title = ttl, saveplot = true)
		end
	end

	return tab
end