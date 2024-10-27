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
    a_grid_small = range(a̲, stop=cut, length=Na_small)
    a_grid_large = range(cut, stop=a̅, length=Na - Na_small)
    a_grid = vcat(a_grid_small, a_grid_large)
    return a_grid
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
    policy = similar(v_array, Int64)

    while i < maxiter && !converged
        v_array_expectation = v_array * P' # E_z_next [V[a_next, z_next] | z] = Σ_z V[a_next, z] * P[z_next | z]
        err = 0.0
        for j in CartesianIndices(v_array)
            a_now_ind, z_now_ind = Tuple(j)
            z_now = z_grid[z_now_ind]
            a_now = a_grid[a_now_ind]
            c_now = @. w * z_now + (1 + r) * a_now - a_grid # c_now is a vector indexed by next period wealth c[a_next]
            utility = @. softlog(c_now) + β * v_array_expectation[:, z_now_ind] # use softlog to avoid log(negative c)
            v, a_next_ind = findmax(utility)
            policy[j] = a_next_ind # store the index of a_next
            v_update[j] = v
            v_old = v_array[j]
            if !isfinite(v) && !isfinite(v_old)
                err += 0.0 # avoid NaN, assume -Inf === -Inf
            else
                err += abs(v - v_old)
            end
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
    r_tmp = 0.04071164238844122

    converged = false
    maxiter = 1000
    i = 0
    stepsize = 0.01

    a_stationary_distribution = zeros(Na)
    z_stationary_distribution = zeros(Nz)
    v_res = zeros(Na, Nz)
    policy_res = zeros(Int64, Na, Nz)

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
            for (j,k) in enumerate(policy_res[:, z_now_ind])
                a_transition_mat_given_z[j, k] = 1.0
            end
            a_transition_mat_given_z * z_transition
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
        @infiltrate
    end

    if !converged
        throw("Did not converge")
    end
    
    policy_res = a_grid[policy_res]

    return r_tmp, w_tmp, a_stationary_distribution, z_stationary_distribution, v_res, policy_res
end

function equi_err(params, r_tmp, tol=1e-6)
    @unpack A, α, δ, a̲, a̅, Na, Nz = params

    a_stationary_distribution = zeros(Na)
    z_stationary_distribution = zeros(Nz)
    v_res = zeros(Na, Nz)
    policy_res = zeros(Int64, Na, Nz)

    lgz_grid, P = tauchen(params)
    z_grid = exp.(lgz_grid)
    a_grid = get_a_grid(params)
    
    v_res, policy_res = vfi(w_tmp, r_tmp, params, v_res, tol)
    joint_transition = BlockArray{Float64}(undef, repeat([Na], Nz), repeat([Na], Nz))
    for (z_now_ind, z_next_ind) in Iterators.product(1:Nz, 1:Nz)
        z_transition = P[z_now_ind, z_next_ind]
        a_transition_mat_given_z = zeros(Na, Na)
        for (j,k) in enumerate(policy_res[:, z_now_ind])
            a_transition_mat_given_z[j, k] = 1.0
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
    K = max(K[1], 0) # avoid precision issue that gives neg capital
    L = L[1]
    r_new = α * A * (K/L)^(α-1) - δ
    err = abs(r_new - r_tmp) + abs(w_new - w_tmp)

    @infiltrate
    println("err = $err, r = $r_tmp, w = $w_tmp")
    return err, a_stationary_distribution, z_stationary_distribution, v_res, policy_res
end

aiyagari_params = ExoParams(β=0.96, δ=0.05, α=0.36, A=1.0, σ=0.2, ρ=0.9, Nz=7, Na=200, a̲=0.0, a̅=20.0, m=3.0)

# loss_func(x) = equi_err(aiyagari_params, exp(x[1]), exp(x[2]))[1]
# res = optimize(loss_func, log.([1.420362563698285, 0.03823852277855579]), show_trace=true)
# _, a_stationary_distribution, z_stationary_distribution, v_res, policy_res = equi_err(aiyagari_params, exp(res.minimizer[1]), exp(res.minimizer[2]))

r = 0.038219210986448225

r, w, a_stationary_distribution, z_stationary_distribution, v, policy = naive_solver(aiyagari_params)
# _, a_stationary_distribution, z_stationary_distribution, v_res, policy_res = equi_err(aiyagari_params, w, r, 1e-6)
# using Plots

# bar(a_stationary_distribution, title="Stationary Distribution of Assets", xlabel="Asset Grid Index", ylabel="Probability")