using ForwardDiff
using Plots
using CPUTime

using StaticArrays
using ArraysOfArrays


function split_vec_of_arrays(u)
    vec.(u) |>
    x ->
    VectorOfSimilarVectors(x).data |>
    transpose |>
    VectorOfSimilarVectors
end


const Q = [
    3.0 -1.0
    -1.0 1.0
]

const b = [
    -2.0
    0.0
]

const x₀ = [
    -4.0
    -3.0
]

const x_star = [
    1.0
    1.0
]

const J_min = -1.0

"""評価関数"""
J(x) = (1/2 .* x' * Q * x .+ b' * x)[1]



"""再急降下法"""
function gradient_descent(α)
    ispan = 1000
    x = Vector{typeof(x₀)}(undef, ispan)
    x[1] = x₀

    for i in 2:ispan
        x[i] = x[i-1] + α * -ForwardDiff.gradient(J, x[i-1])
    end

    return x
end


"""ニュートン法"""
function newton()
    ispan = 1000
    x = Vector{typeof(x₀)}(undef, ispan)
    x[1] = x₀

    for i in 2:ispan
        x[i] = x[i-1] - (inv(ForwardDiff.hessian(J, x[i-1])) * ForwardDiff.gradient(J, x[i-1]))
    end

    return x

end



"""レポ用"""
function main()
    x1 = -5.0:0.01:5.0
    x2 = -5.0:0.01:5.0

    z = [J([_x1, _x2])[1] for _x1 in x1, _x2 in x2]'
    fig1 = plot(
        x1, x2, z, st=:surface, camera=(70,60),
        xlabel="x1", ylabel="x2", zlabel="J"
    )
    fig2 = plot(
        x1, x2, z,
        xlabel="x1", ylabel="x2", zlabel="J"
    )
    scatter!(fig1, [x_star[1]], [x_star[2]], [J_min], label="x*")
    scatter!(fig2, [x_star[1]], [x_star[2]], [J_min], label="x*")


    # 再急降下法
    α_range = (0.01, 0.1, 0.5)
    
    for α in α_range
        x = gradient_descent(α)
        x1, x2 = split_vec_of_arrays(x)
        z = J.(x)
        plot!(fig1, x1, x2, z, label="gradient descent a = " * string(α))
        plot!(fig2, x1, x2, z, label="gradient descent a = " * string(α))
    end

    # ニュートン法
    x = newton()
    x1, x2 = split_vec_of_arrays(x)
    z = J.(x)
    plot!(fig1, x1, x2, z, label="newton")
    plot!(fig2, x1, x2, z, label="newton")


    plot!(fig2, camera=(0, 90))

    return fig1, fig2
end

#x = newton()
@time fig1, fig2, = main()
fig2