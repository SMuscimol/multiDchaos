module AdaptiveNet

using MultivariateStats
#using PyPlot
#import PyPlot.plot
using DSP
using StatsBase
using Optim
using JLD
try
	using QuadGK
catch
end
using AbstractFFTs
using SpecialFunctions

include("power_spectrum.jl")
include("eigenvalues.jl")
include("meanfield.jl")
#include("Qfactor.jl")

end