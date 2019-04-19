function run_benchmark(f::Function, problem::Symbol, method::Symbol = :none; max_evals::Integer = 1)
	t     = Float64[]
	evals = Int64[]
	stats = Dict{Symbol, Any}()

	map(1:max_evals) do i
		out_i, t_i, = @timed f()
		push!(t, t_i)

		if method != :none
			stats_i, evals_i = out_i[end-1],  out_i[end]
			if method == :try
				push!(evals, sum(values(evals_i)))
			else
				push!(evals, evals_i[method])
			end

			for (key, val) in stats_i
				get!(stats, key, typeof(val)[])
				push!(stats[key], val)
			end
		end
	end


	## Create DataFrame with results
	if problem in [:l1, :lInf, :philpott]
		DataFrame(label = "$problem", t = mean(t), tstd = std(t), evals = NaN, evalsstd = NaN,
				  x = NaN, xstd = NaN, lb = NaN, lbstd = NaN, ub = NaN, ubstd = NaN)
	elseif problem in [:mod1, :mod2, :mod3, :mod4, :l2]
		if problem == :mod3
			key = :λ
		else
			key = :μ
		end

		DataFrame(label = "$(problem)_$(method)",
				  t = mean(t),           tstd = std(t),
				  evals = mean(evals),   evalsstd = std(evals),
				  x = mean(stats[key]),  xstd = std(stats[key]),
				  lb = mean(stats[:lb]), lbstd = std(stats[:lb]),
				  ub = mean(stats[:ub]), ubstd = std(stats[:ub]))
	end
end


## --------------------------------------------------------------------------------------------------------
## table
function timecomparison(f::Function, n)
    out, t = @timed f(n)
    return t
end


function timecomparison(f::Function, ns::AbstractArray; maxreps::Integer = 10)

    ## precompilation
    [f(minimum(ns)) for k in 1:3]

    ts = map(ns) do n
        t = map((rep) -> timecomparison(f, n), 1:maxreps)
        return [mean(t) std(t)]
    end
    T = reduce(vcat, ts)
    return T[:,1], T[:,2]
end


## --------------------------------------------------------------------------------------------------------
## table
function maketable(N, n, m, tabs::DataFrame...; savename::String = "")
	tab     = vcat(tabs...)
	labels  = sort(unique(tab[:label]))
	columns = setdiff(names(tab), [:label])

	## create final table
	table = map(labels) do label
		ind = tab[:label] .== label
		df  = tab[ind, 2:end]
		names!(df, Symbol.(names(df), "_", label))
		return df
	end |> cols -> reduce(hcat, cols)

	## sort by columns
	c_rep = repeat(columns, inner = length(labels))
	l_rep = repeat(labels,  outer = length(columns))
	columns_sorted = Symbol.(c_rep, "_", l_rep)

	table = hcat(DataFrame(N = N, n = n, m = m), table[columns_sorted])

	## save
	!isempty(savename) && CSV.write(savename, table)
	return table
end


## --------------------------------------------------------------------------------------------------------
## plot
function figure(; legend::Bool = true, kwargs...)
    return plot(legend = legend)
end


function asymmetricstd(t, err)
    ind   = t .- err .<= 1e-8
    err_l = err
    err_l[ind] .= 1e-8

    return hcat(err_l, err)
end


function splitlabels(tab::DataFrame)
	labels  = string.(names(tab))
	splited = [Symbol.(split(l, "_")) for l in labels if occursin("_", l)]
	splited = [length(s) == 2 ? vcat(s, :none) : s for s in splited]
	splited = reduce(hcat, splited)
	return tuple([unique(splited[row, :]) for row in 1:size(splited,1)]...)
end


function check(x, x_set, x_name::String = "x")
	xs = ∩(x, x_set)
	if length(x) != length(xs)
		s1 = string(setdiff(x, xs))[7:end]
		s2 = string(x_set)[7:end]

		@info "$x_name = $s1 not defined, use $s2"
	end
	return xs
end


function experiments_plot(tab::DataFrame,
						  vars::Array{<:Symbol},
						  prbs::Array{<:Symbol},
						  mtds::Array{<:Symbol};
						  xscale::Symbol = :identity,
	                      yscale::Symbol = :identity,
	                      title::String  = "title",
	                      path::String   = "",
	                      type::String   = "svg",
						  yerror::Bool = true,
	                      saveplot::Bool = false,
	                      kwargs...)

	variables, problems, methods = splitlabels(tab)

	vs = check(vars, variables, "variable")
	ps = check(prbs, problems, "problem")
	ms = check(mtds, methods, "method")

    f = figure(kwargs...)
    for v in vs, p in ps, m in ms
		if m == :none
			label   = string(p)
			col     = Symbol(v, "_", p)
			col_err = Symbol(v, "std_", p)
		else
			label   = string(p, " ", m)
			col     = Symbol(v, "_", p, "_", m)
			col_err = Symbol(v, "std_", p, "_", m)
		end

		col ∉ names(tab)        && continue
		(m == :none && v != :t) && continue

		x    = tab[:N]
		y    = tab[col]
		yerr = asymmetricstd(x, tab[col_err])

		if yerror
			plot!(x, y, yerror = yerr, label = label, xscale = xscale, yscale = yscale)
		else
			plot!(x, y, label = label, xscale = xscale, yscale = yscale)
		end
    end

    title!(split(title, "/")[end])
    xlabel!("n")
    ylabel!(join(string.(vs), ", "))

    saveplot && savefig(f, joinpath(path, string(title, ".", type)))
    return f
end


function mu_plot(tab::DataFrame,
				 mtds::Array{<:Symbol};
                 title::String  = "title",
                 path::String   = "",
                 type::String   = "svg",
                 saveplot::Bool = false,
                 kwargs...)

	f = figure(kwargs...)
	for mtd in mtds
		plot!(tab[Symbol("x_", mtd)], tab[Symbol("f_", mtd)], label = string(mtd))
	end

	lb = minimum([minimum(tab[Symbol("x_", mtd)]) for mtd in mtds])
	ub = maximum([maximum(tab[Symbol("x_", mtd)]) for mtd in mtds])
	plot!([lb, ub], [0, 0], linestyle = :dash, color = :red, label = "0")
	xlabel!("mu")
	xlabel!("f(mu)")

	saveplot && savefig(f, joinpath(path, string(title, ".", type)))
end
