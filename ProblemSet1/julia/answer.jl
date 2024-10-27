using Plots
using Distributions
using Integrals
using Optim
using NLsolve
using Roots
using LinearAlgebra
using Infiltrator
using Flux

function intensive_labor_supply(w, z, a, r, τ, T, η, χ, β; solver=:iterative)
    c_init = [z*w*(1-τ)/η]
    
    if solver == :bisection
        function to_solve(c)
            RHS = (z*w*(1-τ)/(η*c))^χ * z*w*(1-τ) + a*(1+r*(1-τ)) + T
            return RHS/(1+β) - c
        end
        c = 0.0
        lb = (a*(1+r*(1-τ)) + T)/(1+β)
        try
            c = find_zero(to_solve, (lb, 1e+8), Bisection())
        catch e
            @infiltrate
        end
        n = (z*w*(1-τ)/(c*η))^χ
        return c, n
    elseif solver == :iterative
        stepsize = 0.1
        c_tmp = c_init[1]
        iter = 0
        is_converged = false
        while iter < 1000
            RHS = (z*w*(1-τ)/(η*c_tmp))^χ * z*w*(1-τ) + a*(1+r*(1-τ)) + T
            if RHS < 0
                c_tmp = c_tmp * 0.5 # halve it and try again
                continue
            end
            c_new = RHS/(1+β)
            err = abs(c_new - c_tmp)
            if err < 1e-5
                is_converged = true
                break
            else
                # println("err = ", err)
                c_tmp = stepsize * c_new + (1-stepsize) * c_tmp
            end
            iter += 1
        end
        if !is_converged
            println("w = ", w, " z = ", z, " T = ", T, " η = ", η, " χ = ", χ, " β = ", β)
            throw("Convergence failed")
        end
        n = (z*w*(1-τ)/(c_tmp*η))^χ
        return c_tmp, n
    end
end

function N(a, b, r, τ, T, β)
    c = (b + a*(1+r*(1-τ)) + T)/(1+β)
    a_prime = β*c
    return log(c) + β*log(a_prime)
end

function W(c_equi, n_equi, η, χ, β)
    a_prime = β*c_equi
    return log(c_equi) - η*n_equi^(1+1/χ)/(1+1/χ) + β*log(a_prime)
end

function individual_labor_supply(w, z, a, b, r, τ, T, η, χ, β)
    c_ifwork, n_ifwork = intensive_labor_supply(w, z, a, r, τ, T, η, χ, β)
    N_val = N(a, b, r, τ, T, β)
    W_val = W(c_ifwork, n_ifwork, η, χ, β)
    if W_val > N_val
        return c_ifwork, n_ifwork, 1.0
    else
        c = (b + a*(1+r*(1-τ)) + T)/(1+β)
        return c, 0.0, 0.0
    end
end

function aggregate_label_supply_quad_approx(w, a, b, r, τ, T, η, χ, β, z̄, σ_z)
    z_dist = LogNormal(log(z̄), σ_z)
    labor_supply = solve(IntegralProblem((x,p) -> begin
        pdf(z_dist, x) * x * individual_labor_supply(w, x, a, b, r, τ, T, η, χ, β)[2]
    end, (0.0, Inf)), QuadGKJL()).u
    employment = solve(IntegralProblem((x,p) -> begin
        pdf(z_dist, x) * individual_labor_supply(w, x, a, b, r, τ, T, η, χ, β)[3]
    end, (0.0, Inf)), QuadGKJL()).u
    labor_second_moment = solve(IntegralProblem((x,p) -> begin
        pdf(z_dist, x) * (x * individual_labor_supply(w, x, a, b, r, τ, T, η, χ, β)[2])^2
    end, (0.0, Inf)), QuadGKJL()).u
    return labor_supply, employment, sqrt(labor_second_moment - labor_supply^2)
end

function aggregate_label_supply_simulated(w, a, b, r, τ, T, η, χ, β, z̄, σ_z)
    z_dist = LogNormal(log(z̄) - 0.5*σ_z^2, σ_z)
    z_vals = rand(z_dist, 10000)
    labor_supply = map(z_vals) do z
        z * individual_labor_supply(w, z, a, b, r, τ, T, η, χ, β)[2]
    end
    employment = map(z_vals) do z
        individual_labor_supply(w, z, a, b, r, τ, T, η, χ, β)[3]
    end
    return mean(labor_supply), mean(employment), std(labor_supply)
end

function wage_solver(a, b, r, τ, η, χ, β, z̄, σ_z, α, A; verbose=false, stepsize = 0.1)
    w_tmp = 1.0
    T_tmp = 0.0
    iter = 0
    converged = false
    while iter < 1000 
        labor_supply, employment, _ = aggregate_label_supply_quad_approx(w_tmp, a, b, r, τ, T_tmp, η, χ, β, z̄, σ_z)

        if labor_supply < 1e-8
            # don't update it according to firms' FOC
            @info "labor_supply is just so small... increasing wage until it's not crazily small"
            w_tmp = 2.0 * w_tmp # raise it until labor_supply > 1e-8
            continue
        end

        w_new = (1-α)*A*a^α*labor_supply^(-α)
        T_new = (w_tmp * labor_supply + a * r) * τ - (1 - employment) * b
        err = abs(w_new - w_tmp) + abs(T_new - T_tmp)
        
        if err < 1e-4
            converged = true
            break
        else
            if verbose
                println("solver err = ", err, ", w_new = ", w_new)
            end
            w_tmp = stepsize * w_new + (1-stepsize) * w_tmp
            T_tmp = stepsize * T_new + (1-stepsize) * T_tmp
        end
        iter += 1
    end
    if !converged
        while iter < 1000 
            function err(w_)
                labor_supply, employment, _ = aggregate_label_supply_quad_approx(w_, a, b, r, τ, T_tmp, η, χ, β, z̄, σ_z)
                w_new = (1-α)*A*a^α*labor_supply^(-α)
                w_ - w_new
            end
            w_tmp = find_zero(err, (0.01, 2.0), Bisection())
            labor_supply, employment, _ = aggregate_label_supply_quad_approx(w_tmp, a, b, r, τ, T_tmp, η, χ, β, z̄, σ_z)
            T_new = (w_tmp * labor_supply + a * r) * τ - (1 - employment) * b
    
            err = abs(T_new - T_tmp)
            
            if err < 1e-4
                converged = true
                break
            else
                if verbose
                    println("solver err = ", err, ", w = ", w_tmp)
                end
                # w_tmp = stepsize * w_new + (1-stepsize) * w_tmp
                T_tmp = stepsize * T_new + (1-stepsize) * T_tmp
            end
            iter += 1
        end
    end
    if !converged
        println("wage_solver failed to converge")
    end
    return w_tmp, T_tmp
end

# Exogenous vars
const a = 8.0
const α = 0.3
const τ = 0.15
const z̄ = 1.0
const A = 1.0
const r = 0.04
const β = 0.96
const χ = 1.0
const target = [0.33, 0.06, 0.25, 0.70]
const weight_mat = I(4)

function moments(a, b, r, τ, η, χ, β, z̄, σ_z, α, A)
    println("============================")
    println("unemp benefit   :", b)
    println("disutility labor:", η)
    println("elasticity labor:", χ)
    println("variance of z   :", σ_z)
    println("============================")
    w, T = wage_solver(a, b, r, τ, η, χ, β, z̄, σ_z, α, A)
    effective_labor, employment, labor_std = aggregate_label_supply_quad_approx(w, a, b, r, τ, T, η, χ, β, z̄, σ_z)
    unemp = 1 - employment

    M0 = effective_labor / employment
    M1 = unemp
    M2 = unemp * b / (effective_labor * w)
    M3 = labor_std / effective_labor

    
    println("wage            :", w)
    println("gov. transfer   :", T)
    println("effective_labor :", effective_labor)
    println("unemployment    :", unemp)
    println("moment 0 (0.33) :", M0)
    println("moment 1 (0.06) :", M1)
    println("moment 2 (0.25) :", M2)
    println("moment 3 (0.70) :", M3)
    println("============================")
    return [M0, M1, M2, M3]
end

function loss_func(a, b, r, τ, η, χ, β, z̄, σ_z, α, A, target, weight_mat)
    moment_vec = moments(a, b, r, τ, η, χ, β, z̄, σ_z, α, A)
    err_vec = moment_vec - target
    return err_vec' * weight_mat * err_vec
end

function loss_func_wrapper(x)
    # println("b = ", x[1], ", η = ", x[2], ", χ = ", x[3], ", σ_z = ", x[4])
    # x = [b, η, χ, σ_z]
    x = softplus.(x)
    return loss_func(a, x[1], r, τ, x[2], χ, β, z̄, x[4], α, A, target, weight_mat)
end

initial_x = ones(4)
res = optimize(loss_func_wrapper, [0.0,0.0,0.0,0.0], show_trace=true, iterations=1000)
calibrated = softplus(res.minimizer)
println(calibrated)