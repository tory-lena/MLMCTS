using POMDPs, POMDPModels #, POMDPToolbox
#using MCTS, ParticleFilters

using CSV, Distributions, Combinatorics
using Iterators

#importall POMDPs

struct BasketState
    t::Int64
    x::Float64
    assets::Array{Tuple}
end

mutable struct BasketMDP <: MDP{BasketState, Tuple}
    t0::Int64
    xi::Float64
    dt::Int64  #daily/weekly
    dim::Int64
    Portfolio::Tuple
    discount::Float64
end

BasketMDP(Portfolio)=BasketMDP(100, 1000., 10, length(Portfolio), Portfolio, .95)

isterminal(mdp::BasketMDP, s::BasketState) = (s.t>=250? true : false)

import MCTS.initial_state
initial_state(mdp::BasketMDP, rng::MersenneTwister) = initial_state(mdp)
initial_state(mdp::BasketMDP) =  BasketState(mdp.t0, mdp.xi, estimate_trends(mdp.Portfolio, mdp.t0-1, mdp.t0))


## get variation of all asstes in a Portfolio from the previous day

function get_var(mdp::BasketMDP, t::Int64)
    var=rand(mdp.dim)
    for (i, asset) in zip(1:mdp.dim, mdp.Portfolio)
        var[i]= sum(asset[t-mdp.dt:t]./asset[t-mdp.dt-1:t-1] - 1)
    end
    return var
end

get_var(mdp::BasketMDP, b::BasketState) = get_var(mdp, b.t)

import MCTS.actions
import MCTS.generate_sr

function actions(mdp::BasketMDP)
    A=collect(Iterators.product(Base.Iterators.repeated(0:10:100,mdp.dim)...));
    ix=find(x -> sum(x)<=100, A)
    return A[ix]
end

actions(mdp::BasketMDP, s::BasketState)= actions(mdp)

function basket_l(l::Int64, N::Int64, T::Int64, K::Float64, mu::Array{Float64,1}, sig::Array{Float64,1}, alf::Array{Float64,}; option::Int64=1)
    
    dim = size(mu)[1] #5 #how many stocks?
    Sig = eye(dim) #+  0.25*(ones(dim,dim)-eye(dim)) # correleation of .25 between all assets
    C   = cholfact(Sig)[:L]
    if !isassigned(alf)
        alf = ones(dim,)/dim
    end
    
    #T   = 1  #time step
    Sig = diagm(sig) #diagm([.1, .1, .35, .4, .45])#diagm(0.2+0.05*(1:dim))

    nf = 2^l
    hf = T/nf
    
    #X=zeros(5,N)

    sums = zeros(2,1)

    for N1 = 1:10000:N
        N2 = min(10000,N-N1+1)
        alfm = repmat(alf,1,N2)
        Xf = K*ones(dim,N2)

        if  l==0 
            dWf = C*rand(Normal(0,1), dim,N2)*sqrt(hf) 
            Xf  =  Xf + mu.*Xf*hf + Sig*Xf.*dWf + 0.5*Sig^2*(Xf.*(dWf.^2-hf)) 
            #Xf + r*Xf*hf + Sig*Xf.*dWf + 0.5*Sig^2*(Xf.*(dWf.^2-hf)) #Xf + r*Xf*hf + Sig*dWf 
        else
            for n = 1:nf
                dWf = C*randn(dim,N2)* sqrt(hf)  
                Xf  = Xf + mu.*Xf*hf + Sig*Xf.*dWf + 0.5*Sig^2*(Xf.*(dWf.^2-hf))
                #Xf + (Sig*dWf + repmat(r, 1, N))#r.*Xf*hf + Sig*dWf #Xf + r.*Xf*hf + Sig*Xf.*dWf + 0.5*Sig^2*(Xf.*(dWf.^2-hf))
            end
        end
        #X=Xf
    
        if option==1
            Pf  = alf'*Xf-K
            #max.(0,alf'*Xf-K)
        else
            print("No other optional reward models available")
        end
        #Pf  = exp.(-mu*T)*Pf
        sums = sums +  (sum(Pf)*ones(2,1)).^(1:2)
    end

    #cost = N*nf #cost defined as number of finite timesteps
    return sums #, cost, X
end

function generate_sr(mdp::BasketMDP, s::BasketState, a::Tuple, rng::MersenneTwister)
    mui = [stock[1] for stock=s.assets]; mu=mui[:]
    sigi = [stock[2] for stock=s.assets]; sig=sigi[:]
    
    r=0.
    l=1
    #var=zeros(dim, mdp.dt+1)
    for i=s.t:s.t+mdp.dt
        sums = basket_l(l, 100, 1, 1., mu, sig, collect(a)/100)  #var[:, i-(s.t-1)]=[sum(Xf[j,:])/N-1 for j=1:mpd.dim]
        r+=(sums[1]/100) 
        for (j, stock) in enumerate(Portfolio)
            mu[j], sig[j] = trending(stock, i, 20)   
        end
        any(abs.(mu-mui).>=1.2)? l=6 : l=1
    end
    return BasketState(s.t + mdp.dt, s.x *(1 + r), estimate_trends(mdp.Portfolio, s.t)), s.x*r , l 
end

