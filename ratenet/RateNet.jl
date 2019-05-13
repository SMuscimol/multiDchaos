module RateNet

using MultivariateStats
#using PyPlot
#import PyPlot.plot
using DSP
using StatsBase
using Optim
using JLD
using SparseArrays
using LinearAlgebra

defaultTauA = 100.
defaultC = 1 ./defaultTauA

default_params = Dict(
"g"=>1.,
"theta"=>0.,
"gainType"=>"rectifier",
"tauX"=>1.,
"tauY"=>0.1,
"tauR"=>1.,
"tauA"=>[defaultTauA],
"c"=>[defaultC],
"nAdapt"=>1,
"alpha"=>1.0,
"learn_every"=>2,
"zBias"=>0.,
"adapt_type"=>"subthreshold",
"noisy"=>false,
"noisiness"=>[]
)

include("chaoticNetTypes.jl")
include("runner.jl")
#include("analysis.jl")
include("autocorrelation.jl")

end
