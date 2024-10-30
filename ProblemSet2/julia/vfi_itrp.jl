# the stationary distribution of z is lognormal(0, σ), ∫zdΦ(z) = σ^2. Labor market clearning means that L = σ^2/2
using Interpolations
using UnPack
using Distributions
using Integrals
using Optim
using Statistics
using ProgressMeter
using Infiltrator
using LinearAlgebra
using BlockArrays
using Plots

mutable struct ExoParams
    β::Float64 # discount factor
    δ::Float64 # depreciation rate
    α::Float64 # capital share
    A::Float64 # total factor productivity
    σ::Float64 # standard deviation of technology shock
    ρ::Float64 # persistence of technology shock
    Nz::Int # number of grid points for z
    Na::Int # number of grid points for a
    a̲::Float64 # lower bound of a
    a̅::Float64 # upper bound of a
    m::Float64 # numer of standard deviations for the grid of z
end

ExoParams(; β, δ, α, A, σ, ρ, Nz, Na, a̲, a̅, m) = ExoParams(β, δ, α, A, σ, ρ, Nz, Na, a̲, a̅, m)


function get_price(K, L, params::ExoParams)
    @unpack α, δ, A, σ = params
    r = α * A * (K/L)^(α-1) - δ
    w = (1-α) * A * (K/L)^α
    return r, w
end

function tauchen(params)
    @unpack Nz, ρ, σ, m = params
    μ = 0.0
    z_std = σ
    z_max = μ + m * z_std
    z_min = μ - m * z_std
    z_grid = range(z_min, stop=z_max, length=Nz)
    step = (z_max - z_min) / (Nz - 1)
    P = zeros(Nz, Nz)
    
    for j in 1:Nz
        for k in 1:Nz
            if k == 1
                P[j,k] = cdf(Normal(ρ * z_grid[j], σ*sqrt(1-ρ^2)), z_grid[k] + step / 2)
            elseif k == Nz
                P[j,k] = 1 - cdf(Normal(ρ * z_grid[j], σ*sqrt(1-ρ^2)), z_grid[k] - step / 2)
            else
                P[j,k] = cdf(Normal(ρ * z_grid[j], σ*sqrt(1-ρ^2)), z_grid[k] + step / 2) - 
                         cdf(Normal(ρ * z_grid[j], σ*sqrt(1-ρ^2)), z_grid[k] - step / 2)
            end
        end
    end
    
    return z_grid, P
end

function softlog(x)
    if x > 0
        return log(x)
    else
        return -Inf
    end
end

function get_a_grid(params)
    @unpack Na, a̲, a̅ = params
    Na_small = Na ÷ 3
    cut = 1.0
    a_grid_small = range(a̲, stop=cut, length=Na_small+1)
    a_grid_small = a_grid_small[1:end-1]
    a_grid_large = range(cut, stop=a̅, length=Na - Na_small)
    a_grid = vcat(a_grid_small, a_grid_large)
    return a_grid
end

function equalize_a_grid_dist(params, to_transform, new_Na)
    @unpack Na, a̲, a̅ = params
    a_equal_grid = range(a̲, stop=a̅, length=new_Na)
    a_grid = get_a_grid(params)
    transforma_mat = zeros(Na, new_Na)
    lt(x, y) = x <= y
    for (i, a) in enumerate(a_grid)
        a_next_max_ind = searchsortedfirst(a_equal_grid, a, lt=lt)
        a_next_min_ind = a_next_max_ind - 1
        if a_next_max_ind >= new_Na
            a_next_max_ind = new_Na
            a_next_min_ind = new_Na - 1
        end
        a_next_max = a_equal_grid[a_next_max_ind]
        a_next_min = a_equal_grid[a_next_min_ind]
        a_next_min_ind_weight = (a_next_max - a) / (a_next_max - a_next_min)
        a_next_max_ind_weight = 1 - a_next_min_ind_weight
        transforma_mat[i, a_next_min_ind] = a_next_min_ind_weight
        transforma_mat[i, a_next_max_ind] = a_next_max_ind_weight
    end
    return vec(to_transform' * transforma_mat)
end

function vfi(w, r, params, v_init, tol=1e-6)
    @unpack β, δ, α, A, σ, ρ, Nz, Na, a̲, a̅= params
    lgz_grid, P = tauchen(params)
    z_grid = exp.(lgz_grid)
    a_grid = get_a_grid(params)
    v_array = v_init
    
    maxiter = 1000
    converged = false
    i = 0

    v_update = similar(v_array)
    policy = Matrix{Tuple{Int64, Float64, Int64, Float64}}(undef, Na, Nz)

    while i < maxiter && !converged
        v_array_expectation = v_array * P' # E_z_next [V[a_next, z_next] | z] = Σ_z V[a_next, z] * P[z_next | z]
        # a function of a_next and z_now
        err = 0.0
        nodes = (a_grid, z_grid)
        V_expectation_itp = interpolate(nodes, v_array_expectation, Gridded(Linear()))
        for j in CartesianIndices(v_array)
            a_now_ind, z_now_ind = Tuple(j)
            z_now = z_grid[z_now_ind]
            a_now = a_grid[a_now_ind]
            cash_in_hand = w * z_now + (1 + r) * a_now
            f(a_next_tmp) = softlog(cash_in_hand - a_next_tmp) + β * V_expectation_itp(a_next_tmp, z_now)
            res = optimize(x->-f(x), a̲, min(cash_in_hand, a̅)) # a_next is not in the grid
            if !Optim.converged(res)
                throw("Did not converge")
            end
            a_next = res.minimizer
            v = -res.minimum
            a_next_max_ind = searchsortedfirst(a_grid, a_next)
            a_next_max = a_grid[a_next_max_ind]
            a_next_min_ind = a_next_max_ind - 1
            a_next_min = a_grid[a_next_min_ind]
            a_next_min_ind_weight = (a_next_max - a_next) / (a_next_max - a_next_min)
            a_next_max_ind_weight = 1 - a_next_min_ind_weight
            policy[j] = (a_next_min_ind, a_next_min_ind_weight, a_next_max_ind, a_next_max_ind_weight) # store the index and also the weight
            v_update[j] = v
            v_old = v_array[j]
            err += abs(v - v_old)
        end

        if err < tol
            converged = true
        else
            i += 1
            # println("Iteration $i, error: $err")
            copyto!(v_array, v_update)
        end
    end

    if !converged
        throw("Did not converge")
    end
    
    return v_update, policy
end

function stationary_distribution(P)
    eigvals, eigvecs = eigen(P')
    real_eigvals = real(eigvals)
    stationary = eigvecs[:, argmax(real_eigvals)]
    @assert all(imag(stationary) .≈ 0.0) "Imaginary stationary distribution"
    stationary = real(stationary)
    stationary /= sum(stationary) # Normalize to sum to 1
    return stationary
end

function naive_solver(params, tol=1e-6)
    @unpack A, α, δ, a̲, a̅, Na, Nz = params
    r_tmp = 0.04

    converged = false
    maxiter = 1000
    i = 0
    stepsize = 0.01

    a_stationary_distribution = zeros(Na)
    z_stationary_distribution = zeros(Nz)
    v_res = zeros(Na, Nz)
    policy = Matrix{Tuple{Int64, Float64, Int64, Float64}}(undef, Na, Nz)

    lgz_grid, P = tauchen(params)
    z_grid = exp.(lgz_grid)
    a_grid = get_a_grid(params)
    
    while i < maxiter && !converged
        w_tmp = A*(1-α)*(A*α/(r_tmp + δ))^(α/(1-α))
        v_res, policy_res = vfi(w_tmp, r_tmp, params, v_res, tol)
        joint_transition = BlockArray{Float64}(undef, repeat([Na], Nz), repeat([Na], Nz))
        for (z_now_ind, z_next_ind) in Iterators.product(1:Nz, 1:Nz)
            z_transition = P[z_now_ind, z_next_ind]
            a_transition_mat_given_z = zeros(Na, Na)
            for (j,policy) in enumerate(policy_res[:, z_now_ind])
                a_next_min_ind, a_next_min_ind_weight, a_next_max_ind, a_next_max_ind_weight = policy
                a_transition_mat_given_z[j, a_next_min_ind] = a_next_min_ind_weight
                a_transition_mat_given_z[j, a_next_max_ind] = a_next_max_ind_weight
            end
            joint_transition[Block(z_now_ind), Block(z_next_ind)] = a_transition_mat_given_z * z_transition
        end
        joint_transition = Array(joint_transition)
        az_stationary_distribution = stationary_distribution(joint_transition)
        az_stationary_distribution = reshape(az_stationary_distribution, Na, Nz)
        a_stationary_distribution = sum(az_stationary_distribution, dims=2)
        z_stationary_distribution = sum(az_stationary_distribution, dims=1)
        K = a_stationary_distribution' * a_grid
        L = z_stationary_distribution * z_grid
        K = K[1]
        L = L[1]
        r_new = α * A * (K/L)^(α-1) - δ
        err = abs(r_new - r_tmp)
        if err < tol
            converged = true
        else
            i += 1
            println("Iteration $i, error: $err, r: $r_tmp")
            r_tmp = stepsize * r_new + (1 - stepsize) * r_tmp
        end
    end

    if !converged
        throw("Did not converge")
    end

    policy_res = map(policy_res) do policy
        a_next_min_ind, a_next_min_ind_weight, a_next_max_ind, a_next_max_ind_weight = policy
        a_next_min = a_grid[a_next_min_ind]
        a_next_max = a_grid[a_next_max_ind]
        a_next = a_next_min * a_next_min_ind_weight + a_next_max * a_next_max_ind_weight
        return a_next
    end

    return r_tmp, a_stationary_distribution, z_stationary_distribution, v_res, policy_res
end

function figure_IIa_aiyagari(params)
    @unpack β, A, α, δ, a̲, a̅, Na, Nz = params
    lgz_grid, P = tauchen(params)
    z_grid = exp.(lgz_grid)
    a_grid = get_a_grid(params)
    z_stationary_distribution = stationary_distribution(P)
    L = z_stationary_distribution'z_grid
    r_grid = 0.41
    capital_demand_supply = @showprogress map(r_grid) do r
        K = ((r + δ)/(α * A))^(1/(α-1)) * L
        w = A*(1-α)*(A*α/(r + δ))^(α/(1-α))
        v, policy_res = vfi(w, r, params, zeros(Na, Nz), 1e-6)
        joint_transition = BlockArray{Float64}(undef, repeat([Na], Nz), repeat([Na], Nz))
        for (z_now_ind, z_next_ind) in Iterators.product(1:Nz, 1:Nz)
            z_transition = P[z_now_ind, z_next_ind]
            a_transition_mat_given_z = zeros(Na, Na)
            for (j,policy) in enumerate(policy_res[:, z_now_ind])
                a_next_min_ind, a_next_min_ind_weight, a_next_max_ind, a_next_max_ind_weight = policy
                a_transition_mat_given_z[j, a_next_min_ind] = a_next_min_ind_weight
                a_transition_mat_given_z[j, a_next_max_ind] = a_next_max_ind_weight
            end
            joint_transition[Block(z_now_ind), Block(z_next_ind)] = a_transition_mat_given_z * z_transition
        end
        joint_transition = Array(joint_transition)
        az_stationary_distribution = stationary_distribution(joint_transition)
        az_stationary_distribution = reshape(az_stationary_distribution, Na, Nz)
        a_stationary_distribution = sum(az_stationary_distribution, dims=2)
        Ea = a_stationary_distribution' * a_grid
        @infiltrate
        return K, Ea[1]
    end
    K_vec = [x[1] for x in capital_demand_supply]
    Ea_vec = [x[2] for x in capital_demand_supply]
    inds_to_plot = findall(x->x<20.0, Ea_vec)
    plot(K_vec, r_grid, label="Capital Demand")
    plot!(Ea_vec[inds_to_plot], r_grid[inds_to_plot], label="Capital Supply")
    savefig("figure_IIa_aiyagari.png")
end

aiyagari_params = ExoParams(β=0.96, δ=0.05, α=0.36, A=1.0, σ=0.2, ρ=0.9, Nz=7, Na=200, a̲=0.0, a̅=50.0, m=3.0)
r_tmp, a_stationary_distribution, z_stationary_distribution, v_res, policy_res = naive_solver(aiyagari_params)
# a_stationary_distribution2 = equalize_a_grid_dist(aiyagari_params, a_stationary_distribution, 50)
# bar(a_stationary_distribution2, title="Stationary Distribution of Assets", xlabel="Wealth", ylabel="Density")

# reproduce aiyagari figures
# figure_IIa_aiyagari(aiyagari_params)

function martingale(T, N)
    increments = randn(T, N)
    cumsum(increments, dims=1)
end

function submartingale(T, N)
    increments = max.(randn(T, N),-1.0)
    cumsum(increments, dims=1)
end

function supermartingale(T, N)
    increments = min.(randn(T, N),1.0)
    cumsum(increments, dims=1)
end

function bounded_below_supermartingale(T, N)
    increments = min.(randn(T, N),1.0)
    paths = zeros(T, N)
    for t in 1:T
        if t == 1
            paths[t,:] = increments[t,:]
        else
            paths[t,:] = max.(paths[t-1,:] .+ increments[t,:], -10.0)
        end
    end
    paths
end

plot(martingale(1000, 100), legend=false, title="Martingale")
plot(submartingale(1000, 100), legend=false, title="Submartingale")
plot(supermartingale(1000, 100), legend=false, title="Supermartingale")
plot(bounded_below_supermartingale(1000, 100), legend=false, title="Bounded Below Supermartingale")