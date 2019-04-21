## --------------------------------------------------------------------------------------------------------
## timecomparison
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
## l1, l2 and lInf
function l1_comparison(p_l1, p_l1_solver, sd)
    println("‖p_l1_solver - p_l1‖  = ", round(norm(p_l1_solver .- p_l1), sigdigits = sd));
end

function l1_objective(p_l1, p_l1_solver, c, sd)
    println("L(p_l1_solver) = ", round(c'*p_l1_solver, sigdigits = sd))
    println("L(p_l1)        = ", round(c'*p_l1, sigdigits = sd))
end

function l1_feasibility(p_l1, p_l1_solver, p0, ε, sd)
    println("Solver solution: ")
    println("  ⋅ ∑p = 1:       ", round(sum(p_l1_solver), sigdigits = sd), " = ", 1)
    println("  ⋅ ‖p - p0‖ ≦ ε: ", round(norm(p_l1_solver .- p0, 1), sigdigits = sd), " ≦ ", ε)
    println("Our solution: ")
    println("  ⋅ ∑p = 1:       ", round(sum(p_l1), sigdigits = sd), " = 1")
    println("  ⋅ ‖p - p0‖ ≦ ε: ", round(norm(p_l1 .- p0,1), sigdigits = sd), " ≦ ", ε)
end

function l12Inf_plots(p_l1, p_l2, p_lInf, p0, n)
    ylims = extrema(vcat(p0, p_l1, p_l2, p_lInf)) .+ (-0.01, +0.01)

    p1 = plot(legend = :bottomleft, title = "l1", ylims = ylims)
    scatter!(1:n, p0,   label = "p0", marker = :rect)
    scatter!(1:n, p_l1, label = "p",  marker = :diamond)

    p2 = plot(legend = :bottomleft, title = "l2", ylims = ylims)
    scatter!(1:n, p0,   label = "p0", marker = :rect)
    scatter!(1:n, p_l2, label = "p",  marker = :diamond)

    p3 = plot(legend = :bottomleft, title = "l_infty", ylims = ylims)
    scatter!(1:n, p0,     label = "p0", marker = :rect)
    scatter!(1:n, p_lInf, label = "p",  marker = :diamond)

    display(plot(p1, p2, p3, layout = (1,3), size = (900, 400), fmt = :png))
end

function l12Inf_timecomparison(n; maxreps::Integer = 10)
    function eval_l1(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.minimize_linear_on_simplex_l1(p0, c, ε);
    end;

    function eval_l2(n, method::Symbol)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.minimize_linear_on_simplex_l2(p0, c, ε, method = method);
    end;

    function eval_lInf(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.minimize_linear_on_simplex_lInf(p0, c, ε);
    end;

    function eval_l2_philpott(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.philpott(p0, c, ε);
    end;

    t_l1,           = timecomparison(eval_l1, n; maxreps = maxreps);
    t_l2_newton,    = timecomparison((n) -> eval_l2(n, :newton), n; maxreps = maxreps);
    t_l2_secant,    = timecomparison((n) -> eval_l2(n, :secant), n; maxreps = maxreps);
    t_l2_bisection, = timecomparison((n) -> eval_l2(n, :bisection), n; maxreps = maxreps);
    t_lInf,         = timecomparison(eval_lInf, n; maxreps = 10);

    p1 = plot(legend = false, title = "l1", ylabel = "t [s]")
    plot!(n,  t_l1)

    p2 = plot(legend = :topleft, title = "l2")
    plot!(n, t_l2_newton,    label = "newton")
    plot!(n, t_l2_secant,    label = "secant")
    plot!(n, t_l2_bisection, label = "bisection")

    p3 = plot(legend = false, title = "l_infty")
    plot!(n, t_lInf)

    display(plot(p1, p2, p3, layout = (1,3), size = (900, 400), xlabel = "n", fmt = :png))
end

function philpott_timecomparison(n; maxreps::Integer = 10)

    function eval_l2(n, method::Symbol)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.minimize_linear_on_simplex_l2(p0, c, ε, method = method);
    end;

    function eval_l2_philpott(n)
        p0 = rand(Uniform(0,1), n);
        p0 /= sum(p0);
        c  = rand(Normal(0,1), n);
        ε  = 0.1;
        return Projections.philpott(p0, c, ε);
    end;

    t_l2_philpott, = timecomparison(eval_l2_philpott, n; maxreps = maxreps);
    t_l2,          = timecomparison((n) -> eval_l2(n, :newton), n; maxreps = maxreps);

    plot(xlabel = "n", ylabel = "t [s]", legend = :topleft, size = (700, 400))
    plot!(n, t_l2, label = "l2")
    display(plot!(n, t_l2_philpott, label = "l2 philpott", fmt = :png))
end


## --------------------------------------------------------------------------------------------------------
## mod1
function mod1_comparison(p, p_solver, q, q_solver, r, r_solver, sd)
    println("‖p_solver - p‖ = ", round(norm(p_solver .- p), sigdigits = sd))
    println("‖q_solver - q‖ = ", round(norm(q_solver .- q), sigdigits = sd))
    println("‖r_solver - r‖ = ", round(norm(r_solver .- r), sigdigits = sd))
end

function mod1_objective(p, p_solver, q, q_solver, r, r_solver, p0, q0, r0, sd)
    L(p, p0, q, q0, r, r0) = norm(p - p0)/2 + norm(q - q0)/2 + norm(r - r0)/2
    println("L(p,q,r)                      = ", round(L(p, p0, q, q0, r, r0), sigdigits = sd))
    println("L(p_solver,q_solver,r_solver) = ", round(L(p_solver, p0, q_solver, q0, r_solver, r0), sigdigits = sd))
end

function mod1_feasibility(p, p_solver, q, q_solver, r, r_solver, C1, C2, sd)
    println("Solver solution: ")
    println("  ⋅ ∑p = ∑q:       ", round(sum(p_solver), sigdigits = sd), " = ", round(sum(q_solver), sigdigits = sd))
    println("  ⋅ min(p) ≧ 0:    ", round(minimum(p_solver), sigdigits = sd), " ≧ 0")
    println("  ⋅ min(q) ≧ 0:    ", round(minimum(q_solver), sigdigits = sd), " ≧ 0")
    println("  ⋅ max(p) ≦ C1:   ", round(maximum(p_solver), sigdigits = sd), " ≦ ", C1)
    println("  ⋅ max(q) ≦ C2*r: ", round(maximum(q_solver), sigdigits = sd), " ≦ ", round(C2*r_solver, sigdigits = sd))
    println("Our solution: ")
    println("  ⋅ ∑p = ∑q:       ", round(sum(p), sigdigits = sd), " = ", round(sum(q), sigdigits = sd))
    println("  ⋅ min(p) ≧ 0:    ", round(minimum(p), sigdigits = sd), " ≧ 0")
    println("  ⋅ min(q) ≧ 0:    ", round(minimum(q), sigdigits = sd), " ≧ 0")
    println("  ⋅ max(p) ≦ C1:   ", round(maximum(p), sigdigits = sd), " ≦ ", C1)
    println("  ⋅ max(q) ≦ C2*r: ", round(maximum(q), sigdigits = sd), " ≦ ", round(C2*r, sigdigits = sd))
end

function mod1_plots(p, p_solver, q, q_solver, p0, q0, n, m, sd)
    p1 = plot(legend = :topleft, title = "p vs. p0")
    scatter!(1:n, p0,    label = "p0",       marker = :rect)
    scatter!(1:n, p,     label = "p",        marker = :diamond)

    p2 = plot(legend = :topleft, title = "q vs. q0")
    scatter!(1:m, q0,    label = "q0",       marker = :rect)
    scatter!(1:m, q,     label = "q",        marker = :diamond)

    display(plot(p1, p2, layout = (1,2), size = (900, 400), fmt = :png))
end

function mod1_timecomparison(N; maxreps::Integer = 10)
    function eval_mod1(N, method::Symbol = :newton)
        n  = ceil(Int64, 0.3*N);
        m  = N - n;
        p0 = sort(rand(Normal(0,1), n));
        q0 = sort(rand(Normal(0,1), m));
        r0 = rand(Uniform(0,1))
        C1 = 0.7;
        C2 = 0.9;
        return Projections.simplex_mod1(p0, q0, r0, C1, C2; returnstats = true, method = method);
    end;

    ts, ts_std   = timecomparison((n) -> eval_mod1(n, :secant), N; maxreps = maxreps);
    tb, tb_std   = timecomparison((n) -> eval_mod1(n, :bisection), N; maxreps = maxreps);

    plot(xlabel = "N", ylabel = "t [s]", legend = :topleft, size = (700, 400))
    plot!(N, ts, label = "secant")
    display(plot!(N, tb, label = "bisection", fmt = :png))
end


## --------------------------------------------------------------------------------------------------------
## mod2
function mod2_comparison(p, p_solver, q, q_solver, sd)
    println("‖p_solver - p‖ = ", round(norm(p_solver .- p), sigdigits = sd))
    println("‖q_solver - q‖ = ", round(norm(q_solver .- q), sigdigits = sd))
end

function mod2_objective(p, p_solver, q, q_solver, p0, q0, sd)
    L(p, p0, q, q0) = norm(p - p0)/2 + norm(q - q0)/2
    println("L(p,q)               = ", round(L(p, p0, q, q0), sigdigits = sd))
    println("L(p_solver,q_solver) = ", round(L(p_solver, p0, q_solver, q0), sigdigits = sd))
end

function mod2_feasibility(p, p_solver, q, q_solver, C1, C2, sd)
    println("Solver solution: ")
    println("  ⋅ ∑p = ∑q:          ", round(sum(p_solver), sigdigits = sd), " = ", round(sum(q_solver), sigdigits = sd))
    println("  ⋅ min(p) ≧ 0:       ", round(minimum(p_solver), sigdigits = sd), " ≧ 0")
    println("  ⋅ min(q) ≧ 0:       ", round(minimum(q_solver), sigdigits = sd), " ≧ 0")
    println("  ⋅ max(p) ≦ C1:      ", round(maximum(p_solver), sigdigits = sd), " ≦ ", C1)
    println("  ⋅ max(q) ≦ 1/C2*∑q: ", round(maximum(q_solver), sigdigits = sd), " ≦ ", round(sum(p_solver)/C2, sigdigits = sd))
    println("Our solution: ")
    println("  ⋅ ∑p = ∑q:          ", round(sum(p), sigdigits = sd), " = ", round(sum(q), sigdigits = sd))
    println("  ⋅ min(p) ≧ 0:       ", round(minimum(p), sigdigits = sd), " ≧ 0")
    println("  ⋅ min(q) ≧ 0:       ", round(minimum(q), sigdigits = sd), " ≧ 0")
    println("  ⋅ max(p) ≦ C1:      ", round(maximum(p), sigdigits = sd), " ≦ ", C1)
    println("  ⋅ max(q) ≦ 1/C2*∑q: ", round(maximum(q), sigdigits = sd), " ≦ ", round(sum(p)/C2, sigdigits = sd))
end

function mod2_plots(p, p_solver, q, q_solver, p0, q0, n, m, sd)
    p1 = plot(legend = :topleft, title = "p vs. p0")
    scatter!(1:n, p0,    label = "p0",       marker = :rect)
    scatter!(1:n, p,     label = "p",        marker = :diamond)

    p2 = plot(legend = :topleft, title = "q vs. q0")
    scatter!(1:m, q0,    label = "q0",       marker = :rect)
    scatter!(1:m, q,     label = "q",        marker = :diamond)

    display(plot(p1, p2, layout = (1,2), size = (900, 400), fmt = :png))
end

function mod2_timecomparison(N; maxreps::Integer = 10)
    function eval_mod2(N, method::Symbol = :newton)
        n  = ceil(Int64, 0.3*N);
        m  = N - n;
        p0 = sort(rand(Normal(0,1), n));
        q0 = sort(rand(Normal(0,1), m));
        C1 = 0.7;
        C2 = ceil(Int64, m/10);
        return Projections.simplex_mod2(p0, q0, C1, C2; method = method);
    end;

    ts, ts_std   = timecomparison(n -> eval_mod2(n, :secant),    N; maxreps = maxreps);
    tb, tb_std   = timecomparison(n -> eval_mod2(n, :bisection), N; maxreps = maxreps);

    plot(xlabel = "N", ylabel = "t [s]", legend = :topleft, size = (700, 400))
    plot!(N, ts, label = "secant")
    display(plot!(N, tb, label = "bisection", fmt = :png))
end