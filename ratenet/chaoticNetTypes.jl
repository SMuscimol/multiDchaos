mutable struct SimpleNet
    N :: Int64
    x :: Array{Float64,2}
    r :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    SimpleNet(N, x, r, w, params; intVars=Array([:x])) = new(N, x, r, w, params, intVars)
end

mutable struct Net
    N :: Int64
    x :: Array{Float64,2}
    r :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    z :: Array{Float64,2}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}    
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    Net(N, x, r, w, z, wRO, wF, params; intVars=Array([:x])) = new(N, x, r, w, z, wRO, wF, params, intVars)
end

mutable struct AdaptNet
    N :: Int64
    x :: Array{Float64,2}
    a :: Array{Float64,2}
    r :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    z :: Array{Float64,2}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}    
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    AdaptNet(N, x, a, r, w, z, wRO, wF, params; intVars=Array([:x,:a])) = new(N, x, a, r, w, z, wRO, wF, params, intVars)
end

mutable struct SynFilterNet
    N :: Int64
    x :: Array{Float64,2}
    y :: Array{Float64,2}
    r :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    z :: Array{Float64,2}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}    
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    SynFilterNet(N, x, y, r, w, z, wRO, wF, params; intVars=Array([:x,:y])) = new(N, x, y, r, w, z, wRO, wF, params, intVars)
end

AllNet = Union{SimpleNet, Net, SynFilterNet, AdaptNet}

mutable struct RNet
    N :: Int64
    r :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    z :: Array{Float64,2}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}    
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    RNet(N, r, w, z, wRO, wF, params; intVars=Array([:r])) = new(N, r, w, z, wRO, wF, params, intVars)
end

mutable struct AdaptRNet
    N :: Int64
    a :: Array{Float64,2}
    r :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    z :: Array{Float64,2}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}    
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    AdaptRNet(N, a, r, w, z, wRO, wF, params; intVars=Array([:r,:a])) = new(N, a, r, w, z, wRO, wF, params, intVars)
end

AllRNet = Union{RNet, AdaptRNet}



mutable struct Tilo2Net
    N :: Int64
    h :: Array{Float64,2}
    x :: Array{Float64,2}
    r :: Array{Float64,2}
    z :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    Tilo2Net(N, h, x, r, z, w, wRO, wF, params; intVars=Array([:h,:x])) = new(N, h, x, r, z, w, wRO, wF, params, intVars)
end

mutable struct Tilo3Net
    N :: Int64
    dt :: Float64
    h :: Array{Float64,2}  
    x :: Array{Float64,2}
    y :: Array{Float64,2}
    r :: Array{Float64,2}  # the second dimension here is not useless, contains the temporal information necessay to keep track of the refractory period
    z :: Array{Float64,2}
    w :: Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    wRO :: Array{Float64,2}
    wF :: Array{Float64,2}
    params :: Dict{String,Any}
    intVars :: Array{Symbol,1}
    Tilo3Net(N, dt, h, x, y, r, z, w, wRO, wF, params; intVars=Array([:h,:x, :y])) = new(N, dt, h, x, y, r, z, w, wRO, wF, params, intVars)
    Tilo3Net(N, dt, params) = new(N, dt, zeros(N,1), zeros(N,1), zeros(N,1), zeros(N, 1+ max( round(Int, params["D"]./dt) , round(Int, params["tRef"]./dt))), 
                                  zeros(1,1), zeros(N,N), zeros(N,1), zeros(N,1), params, Array([:h, :x, :y]) )
end


nonAdaptiveNets = Union{Net, SynFilterNet, RNet, Tilo2Net}
adaptiveNets = Union{AdaptNet, AdaptRNet}
AllAllNet = Union{AllRNet,AllNet, Tilo2Net, Tilo3Net}

mutable struct ForceVars
    P :: Array{Float64,2}
    k :: Array{Float64,2}
    rPr :: Array{Float64,2}
    c :: Array{Float64,2}
    e :: Array{Float64,2}
end

mutable struct SimpleMonitor
    x :: Array{Float64,2}
    varSampleInterval :: Int64
    SimpleMonitor(x; varSampleInterval=1) = new(x, varSampleInterval)
end

mutable struct Monitor
    x :: Array{Float64,2}
    z :: Array{Float64,2}
    wRO :: Array{Float64,3}
    varSampleInterval :: Int64
    wSampleInterval :: Int64
    Monitor(N,tRange; varSampleInterval=1, wSampleInterval=1) = new( 
        zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) ,
        zeros(1, round(Int,size(tRange,1)./varSampleInterval)+1) , 
        zeros(N,1, round(Int,size(tRange,1)*1 ./wSampleInterval)+1) ,
        varSampleInterval ,
        wSampleInterval )
end

mutable struct RateMonitor
    r :: Array{Float64,2}
    z :: Array{Float64,2}
    wRO :: Array{Float64,3}
    varSampleInterval :: Int64
    wSampleInterval :: Int64
    RateMonitor(N,tRange; varSampleInterval=1, wSampleInterval=1) = new(
        zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) ,
        zeros(1, round(Int,size(tRange,1)./varSampleInterval)+1) , 
        zeros(N,1, round(Int,size(tRange,1)*1 ./wSampleInterval)+1) ,
        varSampleInterval ,
        wSampleInterval )
end

mutable struct AdaptMonitor
    mon :: Union{Monitor,RateMonitor}
    a :: Array{Float64,3}
    varSampleInterval :: Int64
    AdaptMonitor(N,tRange;nAdapt=1, varSampleInterval=1, wSampleInterval=1) =
        new(
            Monitor(N,tRange; varSampleInterval=varSampleInterval,
                wSampleInterval=wSampleInterval),
            zeros(N,nAdapt,round(Int,size(tRange,1)./varSampleInterval)+1),
            varSampleInterval)
    AdaptMonitor(net::AdaptRNet,tRange; nAdapt=1, varSampleInterval=1, 
        wSampleInterval=1) = new(
            RateMonitor(net.N,tRange;varSampleInterval=varSampleInterval,
                wSampleInterval=wSampleInterval),
            zeros(net.N,nAdapt,round(Int,size(tRange,1)./varSampleInterval)+1),
            varSampleInterval)
end

mutable struct SynFilterMonitor
    mon :: Union{Monitor,RateMonitor}
    y :: Array{Float64,2}
    varSampleInterval :: Int64
    SynFilterMonitor(N,tRange; varSampleInterval=1, wSampleInterval=1) =
        new(
            Monitor(N,tRange; varSampleInterval=varSampleInterval,
                wSampleInterval=wSampleInterval),
            zeros(N, round(Int,size(tRange,1)./varSampleInterval)+1),
            varSampleInterval)
    SynFilterMonitor(net::AdaptRNet,tRange; varSampleInterval=1, 
        wSampleInterval=1) = new(
            RateMonitor(net.N,tRange;varSampleInterval=varSampleInterval,
                wSampleInterval=wSampleInterval),
            zeros(net.N,round(Int,size(tRange,1)./varSampleInterval)+1),
            varSampleInterval)
end

mutable struct Tilo2Monitor
    h :: Array{Float64,2}
    x :: Array{Float64,2}
    r :: Array{Float64,2}
    z :: Array{Float64,2}
    wRO :: Array{Float64,3}
    varSampleInterval :: Int64
    wSampleInterval :: Int64
    Tilo2Monitor(N,tRange; varSampleInterval=1, wSampleInterval=1) = 
        new( zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) ,
             zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) ,
             zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) , 
             zeros(1,round(Int,size(tRange,1)./varSampleInterval)+1) ,
             zeros(N,1,round(Int,size(tRange,1)*1 ./wSampleInterval)+1) ,
             varSampleInterval,
             wSampleInterval )
end

mutable struct Tilo3Monitor
    h :: Array{Float64,2}
    x :: Array{Float64,2}
    y :: Array{Float64,2}
    r :: Array{Float64,2}
    z :: Array{Float64,2}
    wRO :: Array{Float64,3}
    varSampleInterval :: Int64
    wSampleInterval :: Int64
    Tilo3Monitor(N,tRange; varSampleInterval=1, wSampleInterval=1) = new(
        zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) ,
        zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) ,
        zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1),
        zeros(N,round(Int,size(tRange,1)./varSampleInterval)+1) , 
        zeros(1,round(Int,size(tRange,1)./varSampleInterval)+1) ,
        zeros(N,1,round(Int,size(tRange,1)*1 ./wSampleInterval)+1),
        varSampleInterval,
        wSampleInterval )
end

mutable struct totInpMonitor
    
    totInp :: Array{Float64,2}
    totInpMonitor(N,tRange) = new(zeros(N,size(tRange,1)+1))

end

"
Auxiliary variable that is used to integrate the network dynamics.
"
mutable struct Rhs
    x :: Array{Float64,2}
    y :: Array{Float64,2}
    a :: Array{Float64,2}
    r :: Array{Float64,2}
    h :: Array{Float64,2}
    Rhs(net::Union{SimpleNet, Net}) = new( zeros(size(net.x)), zeros(1,1), zeros(1,1), zeros(1,1), zeros(1,1) ) 
    Rhs(net::AdaptNet) = new( zeros(size(net.x)), zeros(1,1), zeros(size(net.a)), zeros(1,1), zeros(1,1))
    Rhs(net::SynFilterNet) = new( zeros(size(net.x)), zeros(size(net.y)), zeros(1,1), zeros(1,1), zeros(1,1) )
    Rhs(net::RNet) = new( zeros(1,1), zeros(1,1), zeros(1,1), zeros(size(net.r)), zeros(1,1))
    Rhs(net::AdaptRNet) = new( zeros(1,1), zeros(1,1), zeros(size(net.a)), zeros(size(net.r)), zeros(1,1))
    Rhs(net::Tilo2Net) = new( zeros(size(net.x)), zeros(1,1), zeros(1,1), zeros(1,1), zeros(size(net.h)))
    Rhs(net::Tilo3Net) = new( zeros(size(net.x)), zeros(size(net.y)), zeros(1,1), zeros(1,1), zeros(size(net.h)) )
end 
export Rhs

export SimpleNet, Net, SynFilterNet, AdaptNet, RNet, AdaptRNet, Tilo2Net, Tilo3Net, ForceVars,
Monitor, SynFilterMonitor, RateMonitor, AdaptMonitor, SimpleMonitor, Tilo2Monitor, Tilo3Monitor, totInpMonitor,
AllNet, AllRNet, AllAllNet, adaptiveNets, nonAdaptiveNets


