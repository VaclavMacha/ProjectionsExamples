## AATP1
function mu_AATP1(μ::AbstractArray{<:Real},
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

	g_μ(μ) = Projections.find_μ_AATP1(μ, p0, q0, r0, C1, C2)

	return DataFrame(x_AATP1 = μ, f_AATP1 = g_μ.(μ))
end


## AATP2
function mu_AATP2(μ::AbstractArray{<:Real},
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
	g_μ(μ) = Projections.find_μ_AATP2(μ, s, p0, q0, C1, C2)

	return DataFrame(x_AATP2 = μ, f_AATP2 = g_μ.(μ))
end


## DRO l2
function mu_DRO_l2(μ::AbstractArray{<:Real},
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

	g_μ(μ) = Projections.find_μ_DRO(μ, p0, c, ε)

	return DataFrame(x_DROl2 = μ, f_DROl2 = g_μ.(μ))
end
