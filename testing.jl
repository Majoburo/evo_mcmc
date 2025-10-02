using Distributions
using DelimitedFiles

const POPULATION_MAX = 10
const POPULATION_MIN = 1
const ndim = 2

function loglike(data, params)
    testlikelihood = MvNormal([0, 0], [1 0; 0 1])
    return logpdf(testlikelihood, params)
end

@kwdef mutable struct Walker
    propose::Function
    reproduce::Function
    current::Array{Float64}
    logL::Float64
    failures::Int
end

function ask_for_new_point(params, logL)
    println("Current point  : $params")
    println("Current loglike: $logL")
    println("Enter your new point:")
    newparams = vec(readdlm(IOBuffer(readline())))
    return newparams::Array{eltype(params)}
end

function copy_self(w::Walker)
    ww = deepcopy(w)
    ww.failures = 0
    return ww
end

nsamples = 10

fakedata = 0.0
samples = Vector{Float64}[]
human_walker1 = Walker(propose = ask_for_new_point,
                       current = [-1.0, -1.0],
                       logL = loglike(fakedata, [-1.0, -1.0]),
                       failures = 0,
                       reproduce = copy_self
                      )
walkers = [human_walker1]
@assert POPULATION_MAX >= length(walkers) >= POPULATION_MIN
while length(samples) < nsamples
    w = rand(walkers)
    new = w.propose(w.current, w.logL)
    newlogL = loglike(fakedata, new)
    loga = log(rand())
    logH = newlogL - w.logL
    if logH > loga
        # accept the proposal
        println("accepting sample!")
        push!(samples, new)
        w.current = new
        w.logL = newlogL
        # spawn a child of this walker
        if length(walkers) < POPULATION_MAX
            push!(walkers, w.reproduce(w))
            println([ww.current for ww in walkers])
        end
        println("All samples so far: $samples")
    else
        w.failures += 1
        if (w.failures > 5) && (length(walkers) > POPULATION_MIN)
            setdiff!(walkers, w)
        end
    end
end

#=

TODOs:
- mcmc with multiple walkers/ agents
- neural agent walker / proposal
  - inputs: current parameters, loglike, past parameters and likelihood?
  - outputs: proposed parameters
- visualization
- gamify so you can get rejected if your points are bad

=#
