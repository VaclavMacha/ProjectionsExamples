## Simplex mod2
function mu_simplex_mod1(μ::AbstractArray{<:Real},
						 n::Integer,
						 m::Integer,
						 C1::Real,
						 C2::Real;
						 p0_dist = Normal(1/2,1),
						 q0_dist = Normal(1/2,1),
						 r0_dist = Uniform(0,1),
					 	 seed::Integer = 1234,
						 kwargs...)
	Random.seed!(seed);

	p0 = rand(p0_dist, n)
	q0 = rand(q0_dist, m)
	r0 = rand(r0_dist)

	g_μ(μ) = Projections.find_μ_mod1(μ, p0, q0, r0, C1, C2)

	return DataFrame(x_mod1 = μ, f_mod1 = g_μ.(μ))
end


## Simplex mod2
function mu_simplex_mod2(μ::AbstractArray{<:Real},
						 n::Integer,
						 m::Integer,
						 C1::Real,
						 C2::Integer;
						 p0_dist = Normal(0,1),
						 q0_dist = Normal(0,1),
					 	 seed::Integer = 1234,
						 kwargs...)
	Random.seed!(seed);

	p0 = rand(p0_dist, n)
	q0 = rand(q0_dist, m)

	s      = vcat(.- sort(q0; rev = true), Inf)
	g_μ(μ) = Projections.find_μ_mod2(μ, s, p0, q0, C1, C2)

	return DataFrame(x_mod2 = μ, f_mod2 = g_μ.(μ))
end


## Minimize linear_on simplex l2
function mu_l2(μ::AbstractArray{<:Real},
			   n::Integer,
			   ε::Real;
			   p0_dist = Uniform(0,1),
			   c_dist  = Normal(0,1),
			   seed::Integer = 1234,
			   kwargs...)

	Random.seed!(seed);

	p0 = rand(p0_dist, n)
	p0 ./= sum(p0)
	c  = rand(c_dist, n)
	c .-= maximum(c .+ 1)

	g_μ(μ) = Projections.find_μ(μ, p0, c, ε)

	return DataFrame(x_l2 = μ, f_l2 = g_μ.(μ))
end