using POMDPs, POMDPModels, POMDPToolbox
#using MCTS, ParticleFilters

using CSV, Distributions, Combinatorics
using Iterators

importall POMDPs

struct BasketState
    t::Int64
    x::Float64
    assets::Array{Tuple}
end

mutable struct BasketPOMDP <: POMDP{BasketState, Tuple, Tuple}
    t0::Int64
    xi::Float64
    dt::Int64  #daily/weekly
    dim::Int64
    Portfolio::Tuple
    discount::Float64
end

BasketPOMDP(Portfolio)=BasketPOMDP(100, 1000., 1, length(Portfolio), Portfolio, .95)

initial_state(pomdp::BasketPOMDP) =  BasketState(pomdp.t0, pomdp.xi, estimate_trends(pomdp.Portfolio, pomdp.t0-1, pomdp.t0))


#if no look back window is given - take all the data
estimate_trends(Portfolio::Tuple, t::Int64)=estimate_trends(Portfolio, 250, t) 

## get variation of all asstes in a Portfolio from the previous day

function get_var(pomdp::BasketPOMDP, t::Int64)
    var=rand(pomdp.dim)
    for (i, asset) in zip(1:pomdp.dim, pomdp.Portfolio)
        var[i]= asset[t]/asset[t-1]-1
    end
    return var
end

get_var(pomdp::BasketPOMDP, b::BasketState) = get_var(pomdp, b.t)

isterminal(pomdp::BasketPOMDP, s::BasketState) = (s.t>=250? true : false)

function actions(pomdp::BasketPOMDP)
    A=collect(Iterators.product(Base.Iterators.repeated(0:10:100,pomdp.dim)...));
    ix=find(x -> sum(x)<=100, A)
    return A[ix]
end

actions(pomdp::BasketPOMDP, s::BasketState)= actions(pomdp)

## generates random - possible - observations
# IN: 'b' or 's' 
# OUT: Array{Float64,1}

generate_o(pomdp::BasketPOMDP, s::BasketState, flag::Char)= flag=='b'? [rand(Normal(stock[1], stock[2])) for  stock=s.assets] : get_var(pomdp, s.t) 
generate_o(pomdp::BasketPOMDP, s::BasketState, a::Tuple, flag::Char) =  generate_o(pomdp, s, flag)

function generate_sr(pomdp::BasketPOMDP, s::BasketState, a::Tuple, flag::Char)
    #var = get_var(pomdp.Portfolio, s.t) #out: Array{Float64}
    o = generate_o(pomdp, s, flag)
    r = ((s.x/100)*collect(a)')*o   #(var+1)
    if flag == 's'
        return BasketState(s.t + 1, s.x + r, estimate_trends(pomdp.Portfolio, s.t)), r  
    else
        return BasketState(s.t + 1, s.x + r, s.assets), r  
    end
end

generate_sr(pomdp::BasketPOMDP, s::BasketState, a::Tuple, rng::MersenneTwister)=generate_sr(pomdp, s, a, 'b')
rand(rng::MersenneTwister, s::BasketState)=s
iterator(s::BasketState)=s

transition(pomdp::BasketPOMDP, s::BasketState, a::Tuple) = generate_sr(pomdp, s, a, 's')[1]
observation(pomdp::BasketPOMDP, s::BasketState, a::Tuple, sp::BasketState) = generate_o(pomdp, s, a, 's')

function estimate_value(pomdp::BasketPOMDP, s::BasketState, d::Int64)
    X=[sum(rand(Normal(stock[1],stock[2]), d-1)) for stock in s.assets]
    return maximum(X)*s.x/(d-1)
end
reward(pomdp::BasketPOMDP, s::BasketState, a::Tuple, sp::BasketState) = generate_sr(pomdp, s, a, 's')[2]