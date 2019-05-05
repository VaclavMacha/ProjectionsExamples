## timecomparison
function time_comparison(f::Function, n)
    out, t = @timed f(n)
    return t
end


function time_comparison(f::Function, ns::AbstractArray; maxreps::Integer = 10)

    ## precompilation
    [f(minimum(ns)) for k in 1:10]

    ts = map(ns) do n
        t = map((rep) -> time_comparison(f, n), 1:maxreps)
        return [mean(t) std(t)]
    end
    T = reduce(vcat, ts)
    return T[:,1], T[:,2]
end


## --------------------------------------------------------------------------------------------------------
## DRO: l1, l2 and lInf norm
function solution_comparison_DRO_l1(p_l1, p_l1_solver, c, p0, ε; sd::Integer = 4)
    println("‖p_l1_solver - p_l1‖  = ", round(norm(p_l1_solver .- p_l1), sigdigits = sd));
end


function visual_comparison_DRO(p_l1, p_l2, p_lInf, p0, n)
    ylims = extrema(vcat(p0, p_l1, p_l2, p_lInf)) .+ (-0.01, +0.01)

    p1 = plot(legend = :bottomleft, title = "(DRO) with l_1 norm", ylims = ylims)
    scatter!(1:n, p0,   label = "p0", marker = :diamond)
    scatter!(1:n, p_l1, label = "p",  marker = :rect)

    p2 = plot(legend = :bottomleft, title = "(DRO) with l_2 norm", ylims = ylims)
    scatter!(1:n, p0,   label = "p0", marker = :diamond)
    scatter!(1:n, p_l2, label = "p",  marker = :rect)

    p3 = plot(legend = :bottomleft, title = "(DRO) with l_infty norm", ylims = ylims)
    scatter!(1:n, p0,     label = "p0", marker = :diamond)
    scatter!(1:n, p_lInf, label = "p",  marker = :rect)

    display(plot(p1, p2, p3, layout = (1,3), size = (900, 400), fmt = :svg))
end


function time_comparison_DRO(n; maxreps::Integer = 10)
    function eval_l1(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.solve_DRO_l1(p0, c, ε);
    end;

    function eval_l2(n, method::Symbol)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.solve_DRO_l2(p0, c, ε, method = method);
    end;

    function eval_lInf(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.solve_DRO_lInf(p0, c, ε);
    end;

    function eval_philpott(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.solve_philpott(p0, c, ε);
    end;

    t_l1,           = time_comparison(eval_l1, n; maxreps = maxreps);
    t_l2_newton,    = time_comparison((n) -> eval_l2(n, :newton), n; maxreps = maxreps);
    t_l2_secant,    = time_comparison((n) -> eval_l2(n, :secant), n; maxreps = maxreps);
    t_l2_bisection, = time_comparison((n) -> eval_l2(n, :bisection), n; maxreps = maxreps);
    t_lInf,         = time_comparison(eval_lInf, n; maxreps = 10);

    p1 = plot(legend = false, title = "(DRO) with l_1 norm", ylabel = "t [s]")
    plot!(n,  t_l1)

    p2 = plot(legend = :topleft, title = "(DRO) with l_2 norm")
    plot!(n, t_l2_newton,    label = "Newton")
    plot!(n, t_l2_secant,    label = "Secant")
    plot!(n, t_l2_bisection, label = "Bisection")

    p3 = plot(legend = false, title = "(DRO) with l_infty norm")
    plot!(n, t_lInf)

    display(plot(p1, p2, p3, layout = (1,3), size = (900, 400), xlabel = "n", fmt = :svg))
end


function time_comparison_philpott(n; maxreps::Integer = 10)

    function eval_l2(n, method::Symbol)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.solve_DRO_l2(p0, c, ε, method = method);
    end;

    function eval_philpott(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.solve_philpott(p0, c, ε);
    end;

    t_l2_philpott, = time_comparison(eval_philpott, n; maxreps = maxreps);
    t_l2,          = time_comparison((n) -> eval_l2(n, :newton), n; maxreps = maxreps);

    plot(xlabel = "n", ylabel = "t [s]", legend = :topleft, size = (700, 400))
    plot!(n, t_l2, label = "(DRO) Our algorithm")
    display(plot!(n, t_l2_philpott, label = "(DRO) Philpott et al.", fmt = :svg))
end


## --------------------------------------------------------------------------------------------------------
## AATP1 and AATP2
function solution_comparison_AATP1(p, p_solver, q, q_solver, r, r_solver,  p0, q0, r0; sd::Integer = 4)
    println("‖p_solver - p‖ = ", round(norm(p_solver .- p), sigdigits = sd))
    println("‖q_solver - q‖ = ", round(norm(q_solver .- q), sigdigits = sd))
    println("‖r_solver - r‖ = ", round(norm(r_solver .- r), sigdigits = sd), "\n")
end


function solution_comparison_AATP2(p, p_solver, q, q_solver, p0, q0; sd::Integer = 4)
    println("‖p_solver - p‖ = ", round(norm(p_solver .- p), sigdigits = sd))
    println("‖q_solver - q‖ = ", round(norm(q_solver .- q), sigdigits = sd))
end


function visual_comparison_AATP12(p1, q1, p2, q2, p0, q0, n, m)
    plot1 = plot(legend = :topleft, title = "Solution p")
    scatter!(1:n, p0, label = "p0",        marker = :diamond)
    scatter!(1:n, p1, label = "p (AATP1)", marker = :rect)
    scatter!(1:n, p2, label = "p (AATP2)", marker = :star5)

    plot2 = plot(legend = :topleft, title = "Solution q")
    scatter!(1:m, q0, label = "q0",        marker = :diamond)
    scatter!(1:m, q1, label = "q (AATP1)", marker = :rect)
    scatter!(1:m, q2, label = "q (AATP2)", marker = :star5)

    display(plot(plot1, plot2, layout = (1,2), size = (900, 400), fmt = :svg))
end


function time_comparison_AATP12(N; maxreps::Integer = 10)
    function eva_AATP1(N, method::Symbol = :newton)
        n  = ceil(Int64, 0.3*N);
        m  = N - n;
        p0 = sort(rand(Normal(0,1), n));
        q0 = sort(rand(Normal(0,1), m));
        r0 = rand(Uniform(0,1))
        C1 = 0.7;
        C2 = 0.9;
        return Projections.solve_AATP1(p0, q0, r0, C1, C2; returnstats = true, method = method);
    end;

    function eva_AATP2(N, method::Symbol = :newton)
        n  = ceil(Int64, 0.3*N);
        m  = N - n;
        p0 = sort(rand(Normal(0,1), n));
        q0 = sort(rand(Normal(0,1), m));
        C1 = 0.7;
        C2 = ceil(Int64, m/10);
        return Projections.solve_AATP2(p0, q0, C1, C2; method = method);
    end;

    ts1, ts1_std   = time_comparison(n -> eva_AATP1(n, :secant), N; maxreps = maxreps);
    tb1, tb1_std   = time_comparison(n -> eva_AATP1(n, :bisection), N; maxreps = maxreps);

    ts2, ts2_std   = time_comparison(n -> eva_AATP2(n, :secant),    N; maxreps = maxreps);
    tb2, tb2_std   = time_comparison(n -> eva_AATP2(n, :bisection), N; maxreps = maxreps);

    plot(xlabel = "N", ylabel = "t [s]", legend = :topleft, size = (700, 400))
    plot!(N, ts1, label = "(AATP1) Secant")
    plot!(N, tb1, label = "(AATP1) Bisection")
    plot!(N, ts2, label = "(AATP2) Secant")
    display(plot!(N, tb2, label = "(AATP2) Bisection", fmt = :svg))
end
