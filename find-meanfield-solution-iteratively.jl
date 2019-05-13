using JLD
using RateNet
using AdaptiveNet
using MyTools


netType = ARGS[1]
gamma = parse(Float64,ARGS[2])
b = parse(Float64, ARGS[3])
gFactor = parse(Float64, ARGS[4])
inpFreq = parse(Float64, ARGS[5])
inpAmp = parse(Float64, ARGS[6])
dataDir = ARGS[7]

if netType == "standard"
	g = gFactor
elseif netType == "adapt"
	g = gFactor.*getcriticalg(gamma,b)
end

inpArea = 0.25.*(inpAmp).^2


gF(x) = piecewiseLinearGain.(x)
tauA = 1
try
	tauA = 1./gamma
end

iterations = 200

shift = true

dFreq = 0.001
maxFreq = 2.0

dIntSigmaFactor = 200

if shift
    freqRange = -maxFreq:dFreq:maxFreq
else
    freqRange = vcat( 0.:dFreq:maxFreq, -(maxFreq-dFreq):dFreq:-dFreq)
end

inpS = zeros(size(freqRange))
inpS[find(x->abs.(x)==inpFreq, freqRange)] = inpArea./dFreq

S0 = ones(size(freqRange))
SR = AdaptiveNet.iterativemethod(iterations, freqRange,S0, g, gF, AdaptiveNet.PWLnonlinearpass;
      externalInput=true, saveAll=false, externalSpectrum=inpS, dFreq=dFreq, maxFreq=maxFreq, netType=netType, tauA = tauA,  beta=b, tauX=1., stopAtConvergence=false, dIntSigmaFactor=200)


toName = Dict(
    "netType"=>netType,
    "gFactor"=>gFactor,
    "beta"=>b,
    "gamma"=>gamma,
    "inpFreq"=>inpFreq,
    "inpAmp"=>inpAmp,
    "iterations"=>iterations
    )

randomSignature = string(round(rand(),4))
filename = getfilename(string(dataDir,"spectrum_iterative_driven_PWL"),toName,randomSignature,".jld")
try 
	close(f)
end
f = jldopen(filename,"w")
f["freqRange"] = freqRange
f["SR"] = SR
f["inpS"] = inpS
close(f)



