"
Returns the total input to a network
"
function getTotInp(net::nonAdaptiveNets; inp=0.)
    net.w * net.r + net.wF*net.z + inp
end

function getTotInp(net::adaptiveNets; inp=0.)
    net.w * net.r + net.wF*net.z - sum( net.a , dims=2) + inp
end

function getTotInp(net::Tilo2Net; inp=0.)
    net.params["tauM"] .* net.w * net.r + net.params["tauM"] .* net.wF*net.z + inp
end

function getTotInp(net::Tilo3Net; inp=0.)
    net.params["tauM"] .* net.w * reshape(net.r[:,end-round(Int,net.params["D"]./net.dt)],(net.N,1)) + net.params["tauM"] .* net.wF*net.z + inp
end

"
Returns the RHS of the x ODE.
"
function rhsX(net::SimpleNet; inp=0.)
   (-net.x + net.w*net.r + net.wF*net.z + inp).*1 ./net.params["tauX"]
end

function rhsX(net::Net; inp=0.)
    (-net.x + getTotInp(net; inp=inp)).*1 ./net.params["tauX"]
end 

function rhsX(net::SynFilterNet; inp=0.)
    (-net.x + net.y).*1 ./net.params["tauX"]
end

function rhsX(net::AdaptNet; inp=0.)
    (-net.x + getTotInp(net; inp=inp)).*1 ./net.params["tauX"]
end 

"
Returns the RHS of the synaptic filter ODE.
"
function rhsY(net::SynFilterNet; inp=0.)
    (-net.y + getTotInp(net; inp=inp)).*1 ./net.params["tauY"]
end

"
Returns the RHS of the adaptation ODE.
"
function rhsA(net::AdaptNet; inp=0.)
    if net.params["adapt_type"]=="superthreshold"
        tmp = -net.a * Matrix(Diagonal(1 ./(net.params["tauA"]))) + net.r*reshape(net.params["c"],(1,net.params["nAdapt"])) 
    elseif net.params["adapt_type"]=="subthreshold"
        tmp = -net.a * Matrix(Diagonal(1 ./(net.params["tauA"]))) + net.x*reshape(net.params["c"],(1,net.params["nAdapt"]))  
    end
end

function rhsA(net::AdaptRNet, gF::Function; inp=0.)
    -net.a * Matrix(Diagonal(1 ./(net.params["tauA"]))) + net.r*reshape(net.params["c"],(1,net.params["nAdapt"]))
end

"
returns the RHS of the rate ODE.
"
function rhsR(net::RNet, gF::Function; inp=0.)
    (-net.r + gF( net.params["g"]*(getTotInp(net;inp=inp) - net.params["theta"] ))).*1 ./net.params["tauR"]
end

function rhsR(net::AdaptRNet, gF::Function; inp=0.)
    (-net.r + gF( net.params["g"]*( getTotInp(net; inp=inp) - net.params["theta"] ))).*1 ./net.params["tauR"]
end

"
Returns the RHS of the ODE for h (membrane potential) for the Tilo's unit.
"
function rhsH(net::Tilo2Net; inp=inp)
    (-net.h + net.params["mu"] + getTotInp(net;inp=inp)).* 1 ./net.params["tauM"] 
end

function rhsH(net::Tilo3Net; inp=inp)
    (-net.h + net.params["mu"] + getTotInp(net; inp=inp)).*1 ./net.params["tauM"]
end

"
Returns the RHS of the ODE for x (fraction of relative refractoriness) for the Tilo's unit.
"
function rhsX(net::Tilo2Net; inp=inp)
    -(1 ./net.params["tauR"] + net.r[:,1]).*net.x[:,1] + net.r[:,1]
end

function rhsX(net::Tilo3Net; inp=inp)
    -(1 ./net.params["tauR"] + net.r[:,end]./(1-net.y[:,1])).*net.x[:,1] + net.r[:,end]  
end 

"
Returns the RHS of the ODE for y (fraction of absolute refractoriness) for the Tilo's unit.
"
function rhsY(net::Tilo3Net; inp=inp)
    net.r[:,end] - net.r[:,end-round(Int,net.params["tRef"]./dt)]
end

"
Returns an Rhs object filled with the Rhs of all the integration variables
"
function rhs(net::Union{SimpleNet, Net}, gF::Function; inp=0.)
    out = Rhs(net)
    out.x = rhsX(net; inp=inp)
    return out
end

function rhs(net::SynFilterNet, gF::Function; inp=0.)
    out = Rhs(net)
    out.x = rhsX(net; inp=inp)
    out.y = rhsY(net; inp=inp)
    return out
end

function rhs(net::AdaptNet, gF::Function; inp=0.)
    out = Rhs(net)
    out.x = rhsX(net; inp=inp)
    out.a = rhsA(net; inp=inp)
    return out
end

function rhs(net::RNet, gF::Function; inp=0.)
    out = Rhs(net)
    out.r = rhsR(net, gF; inp=inp)
    return out
end

function rhs(net::AdaptRNet, gF::Function; inp=0.)
    out = Rhs(net)
    out.r = rhsR(net, gF; inp=inp)
    out.a = rhsA(net, gF; inp=inp)
    return out
end

function rhs(net::Tilo2Net, gF::Function; inp=0.)
    out = Rhs(net)
    out.h = rhsH(net; inp=inp)
    out.x[:,1] = rhsX(net; inp=inp)
    return out
end

function rhs(net::Tilo3Net, gF::Function; inp=0.)
    out = Rhs(net)
    out.h = rhsH(net; inp=inp)
    out.x[:,1] = rhsX(net; inp=inp)
    out.y[:,1] = rhsY(net; inp=inp)
end

"
Updates the rate when it is not an integration variables.
"
function updateRate(net::Union{SimpleNet,AllNet}, gF::Function)
    net.r = gF( net.params["g"] * (net.x .- net.params["theta"]))
end

function updateRate(net::Tilo2Net)
    net.r[:,1] = net.params["lambda0"].*exp((net.h[:,1]-net.params["vTH"])./net.params["Delta"]) .* (1 - net.x[:,1])
end

function updateRate(net::Tilo3Net)
    net.r = circshift(net.r,(0,-1))
    net.r[:,end] = net.params["lambda0"].*exp((net.h[:,1]-net.params["vTH"])./net.params["Delta"]) .* (1 - net.x[:,1] - net.y[:,1])
end

"
Updates z.
"
function updateZ(net::Union{AllNet,AllRNet, Tilo2Net})
    net.z = transpose(net.wRO) * net.r .+ net.params["zBias"]
end

function updateZ(net::Tilo3Net)
    net.z = transpose(net.wRO) * reshape(net.r[:,end],(net.N,1)) + net.params["zBias"]  # CHECK!
end

"
Updates the integration integration variables of a network, but not the auxiliary ones.
"
function updateIntVars(netEnd::AllAllNet, netStart::AllAllNet, rhs::Rhs, dt::Float64)
    
    for (i,intVar) in enumerate(netStart.intVars)
        if netStart.params["noisy"]
            setfield!(netEnd,intVar, getfield(netStart,intVar) + dt.*getfield(rhs,intVar) +
                sqrt(dt).*netStart.params["noisiness"][i].*randn(size(getfield(netStart,intVar))) )
        else
            setfield!(netEnd,intVar, getfield(netStart,intVar) + dt.*getfield(rhs,intVar))
        end
    end

end

"
Updates the auxiliary variables
"
function updateAuxVars(net::SimpleNet, gF::Function)
    updateRate(net, gF)
end

function updateAuxVars(net::AllNet, gF::Function)
    updateRate(net, gF)
    updateZ(net)
end

function updateAuxVars(net::AllRNet, gF::Function)
    updateZ(net)
end

function updateAuxVars(net::Union{Tilo2Net,Tilo3Net}, gF::Function)
    updateRate(net)
    updateZ(net)
end

"
Function that performs the update of all the variables by integrating using the Runge-Kutta 4 method.
Here input, if nonzero, need to be a (N,2) array, that contains the external input at the actual and at the future timestep, in such a way that
then the midpoint is found by interpolation.
For the moment noisiness is not supporte with RK4.
"
function RK4update(net::AllAllNet, dt::Float64, gF::Function, tmp::AllAllNet, tmpRhs::Rhs; inp = 0.)

    #tmp = deepcopy(net)
    #tmp.wRO = copy(net)
    if net.params["noisy"]==true
        println("WARNING: noisyness seems to be true for this net, but it's not supported by RK4 integration.")
    end

    if size(inp,2)==1
        inp = hcat(inp[:], inp[:], inp[:])
    else
        inp = hcat(inp[:,1], 0.5 .* (inp[:,1] + inp[:,2]), inp[:,2])
    end

    k1 = rhs(net, gF; inp=inp[:,1])
    updateIntVars(tmp, net, k1, dt/2)
    updateAuxVars(tmp, gF)
    k2 = rhs(tmp, gF; inp=inp[:,2] )
    updateIntVars(tmp, net, k2, dt/2)
    updateAuxVars(tmp, gF)
    k3 = rhs(tmp, gF; inp=inp[:,2])
    updateIntVars(tmp, net, k3, dt)
    updateAuxVars(tmp, gF)
    k4 = rhs(tmp, gF; inp=inp[:,3])

    tmpRhs = Rhs(net)
    for intVar in net.intVars
        setfield!(tmpRhs,intVar, getfield(k1,intVar) + 2 .* getfield(k2,intVar) + 2 .* getfield(k3,intVar) + getfield(k4,intVar))
    end
    updateIntVars(net,net,tmpRhs,dt/6)
    updateAuxVars(net, gF)

end


function FEupdate(net::AllAllNet, dt::Float64, gF::Function; inp=0.)
    updateIntVars(net, net, rhs(net, gF; inp=inp), dt)
    updateAuxVars(net, gF)
end

"
Main update function, that updates all the variables of a network.
"
function update(net, dt::Float64, gF::Function, method::String, tmp::AllAllNet, tmpRhs::Rhs; inp=0.)

    if method=="FE"
        FEupdate(net, dt, gF; inp=inp)
    elseif method=="RK4"
        RK4update(net, dt, gF, tmp, tmpRhs; inp=inp)
    end

end






### Functions for FORCE ###

function updateForceVars(r::Array{Float64,2}, z::Array{Float64,2}, fV::ForceVars, f::Float64)

    fV.k = fV.P * r # N,1
    fV.rPr = transpose(r) * fV.k #1,1
    fV.c = 1.0 ./(1.0 + fV.rPr) # 1,1
    fV.P = fV.P - fV.k*(transpose(fV.k).*fV.c) # N,N
    fV.e = z - f # 1,1

end

function updateForceVars(net::AllAllNet,fV::ForceVars, f::Float64)
    
    updateForceVars(net.r, net.z, fV, f)

end

function updateForceVars(net::Tilo3Net, fV::ForceVars, f::Float64)

    updateForceVars(reshape(net.r[:,end],(net.N,1)) , net.z, fV, f)

end

#############################

### Store relevant variables to monitors ###
function storeToMonitor(i::Int64, net::AllNet, monitor::SimpleMonitor; inp=0.)
    
    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)
        monitor.x[:,newI] = net.x[:,1]
    end

end

function storeToMonitor(i::Int64, net::AllNet, monitor::Monitor; inp=0.)
    
    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)   
        monitor.x[:,newI] = net.x[:,1]
        monitor.z[1,newI] = net.z[1,1]
    end
    if mod(i,monitor.wSampleInterval)==0
        iP = round(Int,i./monitor.wSampleInterval)
        monitor.wRO[:,1,iP] = net.wRO[:,1] 
    end

end

function storeToMonitor(i::Int64, net::AllNet, monitor::SynFilterMonitor; inp=0.)

    storeToMonitor(i, net, monitor.mon)
    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)  
        monitor.y[:,newI] = net.y[:,1] 
    end
end

function storeToMonitor(i::Int64, net::AllRNet, monitor::RateMonitor; inp=0.)

    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)  
        monitor.r[:,newI] = net.r[:,1]
        monitor.z[1,newI] = net.z[1,1]
    end
    if mod(i,monitor.wSampleInterval)==0
        iP = round(Int,i./monitor.wSampleInterval)
        monitor.wRO[:,1,iP] = net.wRO[:,1] 
    end

end

function storeToMonitor(i::Int64, net::Union{AdaptNet,AdaptRNet}, monitor::AdaptMonitor; inp=0.)
    
    storeToMonitor(i, net, monitor.mon)
    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)  
        monitor.a[:,:,newI] = net.a[:,:] 
    end

end

function storeToMonitor(i::Int64, net::Tilo2Net, monitor::Tilo2Monitor; inp=0.)

    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)  
        monitor.h[:,newI] = net.h[:,1]
        monitor.x[:,newI] = net.x[:,1]
        monitor.r[:,newI] = net.r[:,1]
        monitor.z[1,newI] = net.z[1,1]
    end

    if mod(i,monitor.wSampleInterval)==0
        iP = round(Int,i./monitor.wSampleInterval)
        monitor.wRO[:,1,iP] = net.wRO[:,1] 
    end
end

function storeToMonitor(i::Int64, net::Tilo3Net, monitor::Tilo3Monitor; inp=0.)

    if mod(i,monitor.varSampleInterval)==0
        newI = round(Int,i./monitor.varSampleInterval)  
        monitor.h[:,newI] = net.h[:,1]
        monitor.x[:,newI] = net.x[:,1]
        monitor.y[:,newI] = net.y[:,1]
        monitor.r[:,newI] = net.r[:,1]
        monitor.z[1,newI] = net.z[1,1]
    end
    if mod(i,monitor.wSampleInterval)==0
        iP = round(Int,i./monitor.wSampleInterval)
        monitor.wRO[:,1,iP] = net.wRO[:,1] 
    end
end   

function storeToMonitor(i::Int64, net::AllAllNet, monitor::totInpMonitor; inp=0.)

    monitor.totInp[:,i] = getTotInp(net;inp=inp)[:,1]

end

################################################

### Initialize before learning ###

function initSim(tRange, net::Net; varSampleInterval=1, wSampleInterval=1)
    
    dt = tRange[2] - tRange[1]
    wSize = round(Int,size(tRange,1)*1 ./wSampleInterval)
    #monitor = Monitor( zeros(net.N,size(tRange,1)+1) , zeros(1, size(tRange,1)+1) , zeros((net.N,1, wSize+1)),wSampleInterval )
    monitor = Monitor(net.N, tRange; 
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)
    storeToMonitor(1, net, monitor)
    return dt, monitor

end

function initSim(tRange, net::AdaptNet; varSampleInterval=1, wSampleInterval=1)
    
    netTmp = Net(net.N, net.x, net.r, net.w, net.z, net.wRO, net.wF, net.params)
    dt, mon = initSim(tRange, netTmp)
    monitor = AdaptMonitor(net.N, tRange; nAdapt = net.params["nAdapt"],
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)
    storeToMonitor(1, net, monitor)
    return dt, monitor

end

function initSim(tRange, net::RNet; varSampleInterval=1, wSampleInterval=1)
    
    dt = tRange[2] - tRange[1]
    wSize = round(Int,size(tRange,1)*1 ./wSampleInterval)
    #monitor = RateMonitor( zeros(net.N,size(tRange,1)+1) , zeros(1, size(tRange,1)+1) , zeros((net.N,1, wSize+1)) ,wSampleInterval)
    monitor = RateMonitor(net.N, tRange; 
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)
    storeToMonitor(1, net, monitor)
    return dt, monitor

end

function initSim(tRange, net::AdaptRNet; varSampleInterval=1, wSampleInterval=1)
    
    netTmp = RNet(net.N, net.r, net.w, net.z, net.wRO, net.wF, net.params)
    dt, mon = initSim(tRange, netTmp)
    monitor = AdaptMonitor(net,tRange; nAdapt=net.params["nAdapt"],
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)
    storeToMonitor(1, net, monitor)
    return dt, monitor

end

function initSim(tRange,net::Tilo2Net; varSampleInterval=1, wSampleInterval=1)

    dt = tRange[2] - tRange[1]
    wSize = round(Int, size(tRange,1)*1 ./wSampleInterval)
    #monitor = Tilo2Monitor( zeros(net.N,size(tRange,1)+1) , zeros(net.N,size(tRange,1)+1), zeros(net.N,size(tRange,1)+1) , 
    #    zeros(1,size(tRange,1)+1) , zeros(net.N,1,wSize+1), wSampleInterval)
    monitor = Tilo2Monitor(net.N,tRange;
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)
    storeToMonitor(1,net,monitor)
    return dt, monitor

end

############################################

### Simulation scripts ###

function runForce(tRange, net::AllAllNet , f::Array{Float64,1}, gF::Function,
    method::String; inp=[], forceInit=[],
    varSampleInterval=1, wSampleInterval=1, verbose=true)

    dt, monitor = initSim(tRange, net;
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)

    #initialize tmp : an auxiliary net used during the RK4 integration
    tmp = deepcopy(net)
    tmpRhs = Rhs(net)

    #initialize external input if not present
    if size(inp,1)==0
        inp = zeros(net.N,size(tRange,1))
    end
    
    # force-specific variables
    if typeof(forceInit)==RateNet.ForceVars
        fV = deepcopy(forceInit)
    else
        fV = ForceVars(1 ./net.params["alpha"] * eye(net.N), zeros(net.N,1), zeros(1,1), zeros(1,1), zeros(1,1) )
    end

    for (i,t) in enumerate(tRange)
        
        if mod(i,100)==0 && verbose
            print("\r t:",t)
        end
        
        if method == "FE" || i == size(tRange,1)
            update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i])
        elseif method == "RK4"
            update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i:i+1])
        end

        if mod(i, net.params["learn_every"])==0
            #update the inverse correlation matrix 
            updateForceVars(net, fV, f[i])
            net.wRO = net.wRO - (fV.e.*fV.c).*fV.k # (1,1)*(1,1)*(N,1) --> (N,1)
            tmp.wRO = copy(net.wRO)
        end

        storeToMonitor(i+1, net, monitor)
    end

    return monitor, fV

end 

export runForce

"
tRange and f should be provided only in ONE PERIOD, if periodic.
"
function runForceStop(tRangePeriod, net::AllAllNet , f::Array{Float64,1},
    gF::Function, method::String; epsilon = 0.01, maxCycles = 20, inp=[],
    forceInit=[], varSampleInterval=1, wSampleInterval=1, verbose=true)

    dt = tRangePeriod[2] - tRangePeriod[1] 

    tmp = deepcopy(net)
    tmpRhs = Rhs(net)

    tRange = 0.:dt: ((tRangePeriod[end]+dt)*maxCycles -dt)
    dt, monitor = initSim(tRange, net;
        varSampleInterval=varSampleInterval, wSampleInterval=wSampleInterval)
    if size(inp,1)==0
        inp = zeros(net.N,size(tRange,1))
    end
    ### force specific variables
    if typeof(forceInit)==RateNet.ForceVars
        fV = deepcopy(forceInit)
    else
        fV = ForceVars(1 ./net.params["alpha"] * eye(net.N), zeros(net.N,1), zeros(1,1), zeros(1,1), zeros(1,1) )
    end
    cycle=0
    dwRO = ones(size(net.wRO))
    wROlast = zeros(net.N,1,round(Int,size(tRangePeriod,1)./net.params["learn_every"])+1)

    while cycle<maxCycles && median(dwRO)>epsilon
        cycle+=1
        for (i,t) in enumerate(tRangePeriod)
            tP = t + (cycle-1)*tRangePeriod[end]
            iP = i + (cycle-1)*size(tRangePeriod,1)
            if mod(i,100)==0 && verbose
                print("\r t:",tP)
            end
            
            if method == "FE" || i == size(tRange,1)
                update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i])
            elseif method == "RK4"
                update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i:i+1])
            end
            

            if mod(i, net.params["learn_every"])==0
                #update the inverse correlation matrix 
                updateForceVars(net, fV, f[i])
                net.wRO = net.wRO - (fV.e.*fV.c).*fV.k # (1,1)*(1,1)*(N,1) --> (N,1)
                tmp.wRO = copy(net.wRO)
                wROlast[:,1,round(Int,i./net.params["learn_every"])+1] = copy(net.wRO)
            end
            #dwRO = net.wRO[:,1] - monitor.wRO[:,1,iP]
            storeToMonitor(iP+1, net, monitor)
            
        end
        #dwRO  = mean(abs(mapslices(diff,monitor.wRO[:,1,((cycle-1)*size(tRangePeriod,1)+1):(cycle*size(tRangePeriod,1))],3)),3)*
        #        1 ./mean(abs(monitor.wRO[:,1,((cycle-1)*size(tRangePeriod,1)+1):(cycle*size(tRangePeriod,1))]))
        dwRO = mean(abs.(mapslices(diff, wROlast[:,1:1,:],3)),3)*1 ./mean(abs.(wROlast))
    end
    converged =false
    cyclesToConverge = maxCycles
    if cycle < maxCycles
        converged = true
        cyclesToConverge = cycle
    end

    return monitor, converged, cyclesToConverge, wROlast, fV
end

export runForceStop

function testForce(tRange, net::AllAllNet, gF::Function, method::String;
    inp=[], varSampleInterval=1, verbose=true)
    
    dt, monitor = initSim(tRange, net; varSampleInterval=varSampleInterval)

    tmp = deepcopy(net)
    tmpRhs = Rhs(net)

    if size(inp,1)==0
        inp = zeros(net.N,size(tRange,1))
    end

    for (i,t) in enumerate(tRange)
        
        if mod(i,100)==0 && verbose
            print("\r t:",t)
        end

        if method == "FE" || i == size(tRange,1)
            update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i])
        elseif method == "RK4"
            update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i:i+1])
        end

        storeToMonitor(i+1, net, monitor)

    end

    return monitor
end

export testForce


function simpleRun(tRange, net::AllAllNet, gF::Function, 
    monitor::Union{Monitor, SynFilterMonitor, RateMonitor, AdaptMonitor, SimpleMonitor,
        Tilo2Monitor},
    method::String; inp=[], verbose=true)

    dt = tRange[2] - tRange[1]

    tmp = deepcopy(net)
    tmpRhs = Rhs(net)

    if size(inp,1)==0
        storeToMonitor(1, net, monitor)
    
        for (i,t) in enumerate(tRange)
            
            if mod(i,100)==0 && verbose
                print("\r t:",t)
            end
            
            if method == "FE" || i == size(tRange,1)
                update( net, dt, gF, method, tmp, tmpRhs; inp=zeros(net.N,1))
            elseif method == "RK4"
                update( net, dt, gF, method, tmp, tmpRhs; inp=zeros(net.N,2))
            end

            storeToMonitor(i+1, net, monitor)

        end
    else

        storeToMonitor(1, net, monitor)
        
        for (i,t) in enumerate(tRange)
            
            if mod(i,100)==0 && verbose
                print("\r t:",t)
            end
            
            if method == "FE" || i == size(tRange,1)
                update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i])
            elseif method == "RK4"
                update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i:i+1])
            end

            storeToMonitor(i+1, net, monitor)

        end
    end

    return monitor

end

export simpleRun

function runMultipleMon(tRange, net::AllAllNet, gF::Function, method::String, monitors::Array{Any,1};inp=[],verbose=true)

    dt = tRange[2] - tRange[1]

    tmp = deepcopy(net)
    tmpRhs = Rhs(net)

    if size(inp,1)==0
        inp = zeros(net.N,size(tRange,1))
    end

    for mon in monitors
        storeToMonitor(1, net, mon; inp=0.)
    end
    
    for (i,t) in enumerate(tRange)
        
        if mod(i,100)==0 && verbose
            print("\r t:",t)
        end
        
        if method == "FE" || i == size(tRange,1)
            update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i])
        elseif method == "RK4"
            update( net, dt, gF, method, tmp, tmpRhs; inp=inp[:,i:i+1])
        end

        for j in 1:size(monitors,1)
            storeToMonitor(i+1, net, monitors[j]; inp=inp[:,i])
        end

    end

    return monitors

end

export runMultipleMon


function runCheck(net::AllAllNet, gF::Function, method::String; inp = [],
    dt=0.1, tMax = 1000., varSampleInterval=1)

    tRange = 0.:dt:tMax
    tmp = deepcopy(net)
    tmpRhs = Rhs(net)
    dt,monitor = initSim(tRange, net; varSampleInterval=varSampleInterval)
    #monitor = simpleRun(tRange, net, gF, monitor, method, tmp, tmpRhs; verbose=false)
    monitor = simpleRun(tRange, net, gF, monitor, method;
        inp=inp, verbose=false)
    return monitor
end
export runCheck

function resetstate!(net::Net, gF::Function)
    net.x[:] = 0.5 .*rand(net.N,1)[:]
    net.r[:] = gF(net.params["g"]*(net.x - net.params["theta"]))
    net.z[1,1] = rand()
end

function resetstate!(net::AdaptNet, gF::Function)
    net.x[:] = 0.5 .*rand(net.N,1)[:]
    net.r[:] = gF(net.params["g"]*(net.x - net.params["theta"]))
    net.a[:] = zeros(size(net.a[:]))[:]
    net.z[1,1] = rand()
end
export resetstate!




### Temporary drafts of functions ###

# function findKs(net, rhs::Function, dt::Float64)

#     # define 4 Knet object, that will serve as buffers for the dependent variables
#     #kS = Array{Any}(4)
#     #[kS[i] = deepcopy(net) for i=1:4]
#     kS = map(i->Knet(zeros(size(net.x)),zeros(size(net.a)),zeros(size(net.r))),1:4)
#     # k2 = Knet(zeros(size(net.x)),zeros(size(net.a)),zeros(size(net.r)))
#     # k3 = Knet(zeros(size(net.x)),zeros(size(net.a)),zeros(size(net.r)))
#     # k4 = Knet(zeros(size(net.x)),zeros(size(net.a)),zeros(size(net.r)))

   
#     rhs(net) # needs to return an object of the same type as k, or just change k?

#     k1 = rhs(var) # --> k1 has the same size of var
#     k2 = rhs(var + dt/2 .* k1)
#     k3 = rhs(var + dt/2 .* k2)
#     k4 = rhs(var + dt .* k3)

#     return k1, k2, k3, k4
# end

# function RK4update(var, rhs::Function, dt::Float64)

#     k1, k2, k3, k4 = findKs(var, rhs, dt)
#     var = var + dt/6 .* (k1 + 2 .*k2 + 2 .*k3 + k4)
#     return var
# end

####################################

### Old Functions ####

# function updateX(net::SimpleNet, dt::Float64; inp=0.)

#     net.x = net.x + dt * (- net.x + net.w * net.r + inp)

# end


# function updateX(net::Net, dt::Float64; inp=0.)

#     #net.x = net.x + dt * (- net.x + net.w * net.r + net.wF*net.z + inp) # early version
#     net.x = net.x + dt * (- net.x + getTotInp(net;inp=inp) )

# end

# function updateX(net::AdaptNet, dt::Float64; inp=0.)

#     #net.x = net.x + dt * (- net.x + net.w * net.r + net.wF*net.z + inp ) #early version
#     net.x = net.x + dt * (- net.x + getTotInp(net;inp=inp) )
#     #net.a = net.a + dt * ( -net.a * diagm(1 ./(net.params["tauA"])) + net.r*reshape(net.params["c"],(1,net.params["nAdapt"]))) #superthreshold!
#     net.a = net.a + dt * ( -net.a * diagm(1 ./(net.params["tauA"])) + net.x*reshape(net.params["c"],(1,net.params["nAdapt"]))) #subthreshold!
#     #net.x = net.x - dt * sum( net.a , 2) 

# end

# function updateRate(net::RNet, dt::Float64, gF::Function; inp=0.)

#     net.r = net.r +dt * (-net.r + gF( net.params["g"]*(getTotInp(net;inp=inp) - net.params["theta"] )))

# end

# function updateRate(net::AdaptRNet, dt::Float64, gF::Function; inp=0.)

#     # CHECK THE ROLE OF NOT BACKUPING THE VARIABLE BEFORE THE UPDATE!
#     net.r = net.r +dt * (-net.r + gF( net.params["g"]*( getTotInp(net; inp=inp) - net.params["theta"] )))
#     net.a = net.a + dt * ( -net.a * diagm(1 ./(net.params["tauA"])) + net.r*reshape(net.params["c"],(1,net.params["nAdapt"])))

# end



# ### Update also the remaining variables ### 
# function updateState(net::SimpleNet, dt::Float64, gF::Function; inp=0., method = "FE")

#     net.x = update(net.x, rhsX, dt; inp=inp)
#     net.r = gF( net.params["g"]*(net.x - net.params["theta"]))

# end

# function updateState(net::AllNet, dt::Float64, gF::Function; inp=0.)
    
#     updateX(net, dt; inp=inp)
#     net.r = gF( net.params["g"] * (net.x - net.params["theta"]))
#     net.z = transpose(net.wRO) * net.r  

# end

# function updateState(net::AllRNet, dt::Float64, gF::Function; inp=0.)

#     updateRate(net, dt, gF; inp=inp)
#     net.z = transpose(net.wRO) * net.r + net.params["zBias"]

# end



### Functions for Tilo's Net --- They need to be adapted to the new integration method ###


# function updateX(net::Tilo2Net, dt::Float64; inp=0.)

#     #net.h = net.h + dt * (-net.h + net.params["mu"] + 
#     #    net.params["tauM"].*net.w * net.r + net.wF*
#     #net.z + inp).* 1 ./net.params["tauM"]  # early version
#     net.h = net.h + dt * (-net.h + net.params["mu"] + 
#         getTotInp(net;inp=inp)).* 1 ./net.params["tauM"] 
#     net.x[:,1] = net.x[:,1] + dt * (-(1 ./net.params["tauR"] + net.r[:,1]).*net.x[:,1] + net.r[:,1])

# end

# function updateX(net::Tilo3Net, dt::Float64; inp=0.)

#     net.h = net.h + dt * (-net.h + net.params["mu"] + getTotInp(net; inp=inp)).*1 ./net.params["tauM"]
#     net.x[:,1] = net.x[:,1] +dt * (-(1 ./net.params["tauR"] + net.r[:,end]./(1-net.y[:,1])).*net.x[:,1] + net.r[:,end])
#     net.y[:,1] = net.y[:,1] +dt * (net.r[:,end] - net.r[:,end-round(Int,net.params["tRef"]./dt)])

# end

# function updateState(net::Tilo2Net, dt::Float64, gF::Function; inp=0)

#     updateX(net, dt; inp=inp)
#     net.r[:,1] = net.params["lambda0"].*exp((net.h[:,1]-net.params["vTH"])./net.params["Delta"]) .* (1 - net.x[:,1]) 
#     net.z = transpose(net.wRO) * net.r + net.params["zBias"]

# end

# function updateState(net::Tilo3Net, dt::Float64, gF::Function; inp=0.)

#     updateX(net, dt; inp=inp)
#     net.r = circshift(net.r,(0,-1))
#     net.r[:,end] = net.params["lambda0"].*exp((net.h[:,1]-net.params["vTH"])./net.params["Delta"]) .* (1 - net.x[:,1] - net.y[:,1]) 
#     net.z = transpose(net.wRO) * reshape(net.r[:,end],(net.N,1)) + net.params["zBias"]  # CHECK!

# end

################################################################

#### END OF FILE ####