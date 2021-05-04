### run dynamics and compute spectrum ###
# ARGS must be provided as arguments to the script. Here are some
# possible defaults
# netType = "adapt"  #

# netType = "adapt"  # network type. Can be "adapt", "standard" or "synFilter"
# gFactor = 1.5  # factor multiplying g critical. The effective g is gFactor * gCritical
# gamma = 0.1  # timescale ratio, see paper
# beta = 1.   # adaptation strength, see paper
# wSeed = 1234  # random seed used to generate random connectivity
# nonlinearity = "tanh"  # gain function, can be "tanh" or "PWL"
# tauY = 1.  # time constant of synaptic filtering, if using netType = "synFilter"
# dataDir = "./"  # where to store data
# intMethod = "RK4"  # integration method, can be "FE" for forward Euler or "RK4" for Runge-Kutta-4

netType = ARGS[1]  # can be "adapt", "nonAdapt" or "synFilter"
gFactor =  parse(Float64,ARGS[2])
gamma = parse(Float64,ARGS[3])
beta = parse(Float64,ARGS[4])
wSeed = parse(Int,ARGS[5])
nonlinearity = ARGS[6]  # can be "tanh" or "PWL"
tauY = parse(Float64,ARGS[7])
dataDir = ARGS[8]
intMethod = ARGS[9]  # can be "FE" or "RK4"


# add an optional argument from where to load data #
useTmpResults = false
try
    tmpResultsDir = parse(Float64,ARGS[10])
    useTmpResults = true
catch
    useTmpResults = false
end

#using StatsBase
using RateNet
using AdaptiveNet
using JLD
using MyTools
using FFTW

tauA = 1 ./ gamma
J = gFactor*getcriticalg(gamma, beta)


toNameVariable = Dict{String,Any}()

c = zeros(1)
c[1,1] = beta/tauA[1]
#c[2,1] = .1 ./tauA[2]

paramsA = Dict(
"g"=>1.,
"tauA"=>Array([tauA]),
"c"=>c,
"alpha"=>1.0,
"learn_every"=>2,
"theta"=>0.,
"nAdapt"=>1,
"zBias"=>0.,
"adapt_type"=>"subthreshold",
"tauX"=>1.,
"tauY"=>tauY,
"noisy"=>false,
"noisiness"=>[]
)

#a = 1.
if nonlinearity=="tanh"
    gF(x) = tanh.(x)
elseif nonlinearity=="PWL"
    function gF(x)
        -1. *(x.<-1) + (x.>=-1).*(x.<=1.).*x + 1. *(x.>1)
        #-1 .*(x.<-a) + (x.<=a).*(x.>=-a).*x/(a) + 1 .*(x.>a)
    end
    #merge!(toNameVariable,Dict("a"=>a))
end

N = 1000
# set the seed to a value to have the same w matrix
using Random
Random.seed!(wSeed)
w = J/sqrt(N)*randn(N,N)
# go back to a random seed
Random.seed!(time_ns())



#w = J*sprandn(N,N,p)
#w = full(w)
wRO = zeros(N,1)
wF = zeros(N,1)
z0 = zeros(1,1)



s0 = 0.5 .*rand(N,1)
r0 = gF(paramsA["g"]*(s0 .- paramsA["theta"]))
a0 = zeros(N,paramsA["nAdapt"])
y0 = zeros(N,1)

blockTime = 100.
nBlocks = 11

if netType == "synFilter"
    dt = 0.1*tauY
    tMax = blockTime
    merge!(toNameVariable,Dict("tauY"=>tauY))
elseif netType == "nonAdapt"
    dt = 0.1
    tMax = blockTime
elseif netType == "adapt"
    dt = 0.1 #ms
    tMax = blockTime
    merge!(toNameVariable,Dict("gamma"=>gamma,"beta"=>beta))
end

tRange = 0.:dt:tMax #
lenT = size(tRange,1) + 1 #time steps
deltaT = 1 #time steps


#create the network!
if netType == "nonAdapt"
    net = Net(N,s0,r0,w,z0,wRO,wF,paramsA);
elseif netType == "adapt"
    net = AdaptNet(N,s0,a0,r0,w,z0,wRO,wF,paramsA);
elseif netType == "synFilter"
    net = SynFilterNet(N,s0,y0,r0,w,z0,wRO,wF,paramsA)
end

randomSignature = string(round(rand(),digits=4))

toNameStable = Dict(
    "netType-"=>netType,
    "N"=>N,
    "J"=>J,
    "tMax"=>tMax,
    "nonlinearity-"=>nonlinearity,
    "wSeed"=>wSeed,
    "intMethod"=>intMethod
    )



#blockTime = 1000.
blockLength = round(Int, blockTime./(dt*deltaT))
SxB = Array{Any}(undef, nBlocks)
traces = Array{Any}(undef, nBlocks)
s1s = Array{Any}(undef, nBlocks)
Sx = []

deltaFreq = 1 ./blockTime #KHz
maxFreq = 1 ./(2 .*dt.*deltaT) #KHz
freqRange = -maxFreq:deltaFreq:(maxFreq-deltaFreq); #KHz


nToSave = rand(1:N,10)
merge!(toNameVariable, Dict("deltaT"=>deltaT))
toName = merge(toNameStable, toNameVariable)

if useTmpResults
    filenames = readdir(tmpResultsDir);
    for (key,val) in toName
        filenames = filter(x->contains(x,string(key,string(val))) , filenames)
        filename = string(dataDir,filenames[1]) # only retain the first one
    end
    fi = jldopen(filename, "r")
    traces = read(fi,"traces")
    read(fi,"Sx")
    SxB = read(fi, "SxB")
    freqRange = read(fi, "freqRange")
    s0 = read(fi, "x00")
    mon.x[:,1] = read(fi, "x01")
    close(fi)
    bStart = find(x->x==false, map( b->isassigned(SxB,b),1:nBlocks) )[1]
else
    filename = getfilename(string(dataDir,"dynamics_series_"),toName,randomSignature,".jld")
    bStart = 1
end

for b=bStart:nBlocks
    println("block no.:",b)
    mon = SimpleMonitor(zeros(N,round(Int, size(tRange,1)./deltaT)+1);
        varSampleInterval=deltaT)
    simpleRun(tRange,net,gF,mon, intMethod;verbose=false);
    SxB[b] = Array{Complex{Float64}}(zeros(size(freqRange,1)))
    for i =1:N
        xFT = dt.*deltaT.*fftshift(fft(mon.x[i,1:size(freqRange,1)]))
        SxB[b] = (i-1)./i .* SxB[b] +
            1 ./i .*1 ./(dt.*deltaT.*size(freqRange,1)).*conj.(xFT).*xFT
    end

    traces[b] = mon.x[nToSave,:]
    s1s[b] = mon.x[:,1]

    if b>1
        Sx = 1 ./(b-1).*sum(SxB[2:b])
    end


    f = jldopen(filename,"w")
    #f["net"] = net
    f["traces"] = traces
    f["Sx"] = Sx
    f["SxB"] = SxB
    f["freqRange"] = freqRange
    f["x00"] = s0
    f["x01"] = mon.x[:,1]
    close(f)

end
