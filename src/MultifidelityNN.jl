module MultifidelityNN

using MultifidelitySamples, Flux

export EmbedDict, embed

EmbedDict = Dict{Type{<:AbstractSummaryInformation}, Chain}()
function _check_embed(::Type{Y}) where Y
    Y in keys(EmbedDict) || error("Define `EmbedDict[$Y] = Flux.Chain(...)` please!")
    return nothing
end
function embed(y::Y) where Y<:AbstractSummaryInformation
    _check_embed(Y)
    return EmbedDict[Y](convert(Vector{Float32}, y))
end
function embed(y::AbstractVector{Y}) where Y<:AbstractSummaryInformation
    _check_embed(Y)
    return EmbedDict[Y](convert(Matrix{Float32}, y))
end

function Base.convert(::Type{Vector{Float32}}, ::Y) where Y<:AbstractSummaryInformation
    error("Define `Base.convert(::Type{Vector{Float32}}, $Y)` please!")
end
function Base.convert(::Type{Matrix{Float32}}, ys::AbstractVector{Y}) where Y<:AbstractSummaryInformation
    reduce(hcat, convert.(Vector{Float32}, ys))
end

struct DynamicIntegrator{M_θx, M_xμ, M_uxx <: Flux.RNNCell}
    θx::M_θx
    xμ::M_xμ
    uxx::M_uxx
end
function (μ::DynamicIntegrator)(θ, y...)
    x = μ.θx(convert(Vector{Float32}, θ))
    for y_i in Iterators.reverse(y)
        u_i = embed(y_i)
        x, _ = μ.uxx(x, u_i)
    end
    out = μ.xμ(x)
    return out[1]
end
(μ::DynamicIntegrator)(T::MultifidelityPathTree) = μ(T.θ, T.y...)
    

function DynamicIntegrator(dimθ::Integer, dimU::Integer, dimX::Integer, dimMU::Integer=1)
    θx = Chain(
        Dense(dimθ, dimX, tanh),
        Dense(dimX, dimX, tanh),
    )
    xμ = Chain(
        Dense(dimX, dimX, tanh),
        Dense(dimX, dimMU, softplus),
    )
    uxx = Flux.RNNCell(dimU, dimX)

    return DynamicIntegrator(θx, xμ, uxx)
end

sqnorm(x) = sum(abs2, x)
function train!(μ::DynamicIntegrator, S::MultifidelitySample, G; epochs::Integer, opt = ADAM(), λ::Real = 0.1, batchsize=64)
    ps = Flux.params(μ.θx, μ.xμ, μ.uxx, values(EmbedDict)...)
    m = sum(length, ps)
    
    cost = cost_functional(μ, G)
    loss = sample -> cost(sample) + (λ / 2m)*sum(sqnorm, ps)

    evalcb() = @show(loss(S))
    throttled_cb = Flux.throttle(evalcb, 5)

    opt_data = Flux.DataLoader(S, batchsize=batchsize, shuffle=true)
    Flux.@epochs epochs Flux.train!(loss, ps, opt_data, opt, cb = throttled_cb)
end


end # module