"
Returns the allowed frequency for an adaptive linear net.
"
function getfrequencylinearnet(gamma,b,g; verbose=false)
    Delta = (gamma^2+1-g^2-2*b*gamma)^2 -4*gamma^2 *((b+1)^2-g^2)
    if Delta < 0
        if verbose
            println("Negative argument of the inner square root")
        end
        return NaN, NaN
    end
    OmegaPlus = 0.5 .*(g^2-1+2*b*gamma-gamma^2 +
        sqrt(Delta) )
    OmegaMinus = 0.5 .*(g^2-1+2*b*gamma-gamma^2 -
        sqrt(Delta) )
    if OmegaPlus < 0
        if verbose
            println("OmegaPlus is negative")
        end
        return NaN, NaN
    end
    freqPlus = 1 ./(2*pi) .* sqrt(OmegaPlus)
    if OmegaMinus<0
        freqMinus = NaN
    else
        freqMinus = 1 ./(2*pi) .* sqrt(OmegaMinus)
    end
    return freqPlus, freqMinus
end
export getfrequencylinearnet

"
Given the value of gamma, returns the value of beta for which the adaptive network autocorrelation obeys a conservative ODE.
If alsoNegative==true, it also return the negative value for beta, which would correspond to having facilitation.
"
function getconservativebeta(gamma; alsoNegative=false)
    betaPositive = -(gamma+1) + sqrt((gamma+1)^2 +1)
    if alsoNegative
        betaNegative = -(gamma+1) - sqrt((gamma+1)^2 +1)
        return betaPositive, betaNegative
    else
        return betaPositive
    end    
end
export getconservativebeta


# function eulerStep(delta,delta1,delta2,delta3,delta0; intRange=-10:0.1:10., dTau=0.1, beta=0.5, tauA=100.,J=1.)
#     # saving values
#     deltaBkp, delta1Bkp, delta2Bkp, delta3Bkp = copy(delta), copy(delta1), copy(delta2), copy(delta3)
#     #return deltaBkp,delta1Bkp, delta2Bkp, delta3Bkp
#     #updating values
#     delta += dTau * delta1Bkp
#     delta1 += dTau * delta2Bkp
#     delta2 += dTau * delta3Bkp
#     #print(deltaBkp,delta1Bkp, delta2Bkp, delta3Bkp)
#     delta3 += 1 ./(tauA^2).* dTau * ((1+tauA^2-2*tauA*beta)*delta2Bkp - (1+beta)^2 * deltaBkp - 
#         J^2 * tauA^2 * fAv(deltaBkp,delta0,phiP;intRange=intRange)*delta2Bkp -
#         J^2 * tauA^2 * fAv(deltaBkp,delta0,phiPP;intRange=intRange)*delta1Bkp^2 +
#         J^2 * fAv(deltaBkp,delta0,phi;intRange=intRange))
#     return delta,delta1,delta2,delta3
# end 

# phi(x) = tanh(x)
# phiP(x) = 1 ./((cosh(x)).^2)
# phiPP(x) = -2*tanh(x).*1 ./((cosh(x)).^2)
# Phi(x) = log(cosh(x))

function eulerStep(vars, rhs::Function, h::Float64; rhsKWargs...)
    return vars + h.*rhs(vars; rhsKWargs...)
end 

function eulerStep!(vars, rhs::Function, h::Float64; rhsKWargs...)
    vars +=  h.*rhs(vars; rhsKWargs...)
end

function RK4step(vars, rhs::Function, h::Float64; rhsKWargs...)
    k1 = rhs(vars; rhsKWargs...)
    k2 = rhs(vars + h/2 .* k1; rhsKWargs...)
    k3 = rhs(vars + h/2 .* k2; rhsKWargs...)
    k4 = rhs(vars + h .* k3; rhsKWargs...)
    return vars + h/6 .* (k1 +2 .*k2 + 2 .*k3 + k4)
end

function RK4step!(vars, rhs::Function, h::Float64; rhsKWargs...)
    k1 = rhs(vars; rhsKWargs...)
    k2 = rhs(vars + h/2 .* k1; rhsKWargs...)
    k3 = rhs(vars + h/2 .* k2; rhsKWargs...)
    k4 = rhs(vars + h .* k3; rhsKWargs...)
    vars[:] = vars[:] + h/6 .* (k1 +2 .*k2 + 2 .*k3 + k4)
end
export RK4step, RK4step!

function RK4integration(initVars, rhs::Function, tRange; rhsKWargs...)
    
    h = tRange[2] - tRange[1]
    vars = copy(initVars)
    out = zeros(size(initVars,1),size(tRange,1)+1)
    out[:,1] = copy(vars)

    for (i,t) in enumerate(tRange)
        rhsKWargs = vcat(rhsKWargs, (:tInd,i))
        out[:,i+1] = RK4step!(vars, rhs, h; tInd=i, rhsKWargs...)
    end

    return out
end
export RK4integration

function assigntableglobalvariables(filenameRoot::String, dataDir; deltaRes=0.01, tableTypes = ["PTANH","TANH","TANHP","TANHPP","TANH_TANHPP"])
    
    allTableFiles = filter(x->(contains(x,filenameRoot) && contains(x,string("deltaRes",string(deltaRes)))),readdir(dataDir))
    filename = []
    for tableType in tableTypes
        filename = filter(x->contains(x,string("table",tableType,"_")),allTableFiles)
        assignonetablevariable(string(dataDir,filename[1]), tableType)
    end
    fi = jldopen(string(dataDir,filename[1]),"r")
    try
        global DELTA_RANGE = read(fi,"deltaRange")
        global DELTA0_RANGE = read(fi,"delta0Range")
    catch
        println("Could not read delta ranges, please define them manually!")
    end
    close(fi)
end 
function assignonetablevariable(filename::String, tabletype::String)
    fi = jldopen(filename,"r")
    if tabletype=="PTANH"
        global TABLE_PTANH = read(fi,"tablePTANH")
    elseif tabletype=="TANH"
        global TABLE_TANH = read(fi,"tableTANH")
    elseif tabletype=="TANHP"
        global TABLE_TANHP = read(fi,"tableTANHP")
    elseif tabletype=="TANHPP"
        global TABLE_TANHPP = read(fi,"tableTANHPP")
    elseif tabletype=="TANH_TANHPP"
        global TABLE_TANH_TANHPP = read(fi,"tableTANH_TANHPP")
    end
    close(fi)
end

export assigntableglobalvariables, TABLE_PTANH, TABLE_TANH, TABLE_TANHP, TABLE_TANHPP, TABLE_TANH_TANHPP, DELTA_RANGE, DELTA0_RANGE   



function fAv(delta,delta0::Float64, func; method="table", correctNaN=false, intRange=-10.:0.1:10., customFunc = [])
    
    dRange = intRange[2]-intRange[1]
    tmp = 0.
    no_table = true
    # no_table = false
    # table = []
    # f = []

    # if func=="tanh"
    #     f = x->tanh.(x)
    #     if method=="table"
    #         table = TABLE_TANH
    #     end
    # elseif func=="Ptanh"
    #     f = x->log.(cosh.(x))
    #     if method=="table"
    #         table = TABLE_PTANH
    #     end
    # elseif func=="tanhP"
    #     f = x-> 1 ./((cosh(x)).^2)
    #     if method=="table"
    #         table = TABLE_TANHP
    #     end
    # elseif func=="tanhPP"
    #     f= x->-2*tanh(x).*1 ./((cosh(x)).^2)
    #     if method=="table"
    #         table = TABLE_TANHPP
    #     end
    # else
    #     f = customFunc
    #     method = "direct"
    # end
    
    
    if delta0 == 0.
        tmp = 0.
    else
        if method=="table"
            try
                tmp = readfromtable(delta, delta0, DELTA_RANGE, DELTA0_RANGE, table)
            catch
                no_table = true
                println("Could not read from table")
            end
            if tmp == NaN && correctNaN == true
                tmp = readfromtable(delta0, delta0, table)
            end
        end
        if method=="direct" || no_table == true
            tmp = 1 ./(2*pi) .* dRange^2 .* sum(map(z->exp.(-0.5 .*z.^2).*sum( exp.(-0.5 .*intRange.^2) .* 
                f.(sqrt(max(0.,delta0-(delta^2)/delta0)).*intRange + delta/sqrt(delta0) .* z) .*
                f.(sqrt(delta0).*z) ), intRange ) ) 
        end
    end
    
    return tmp
    
end

function fAv2(delta,delta0::Float64, func1, func2; method="table", correctNaN=false, intRange=-10.:0.1:10., customFunc1 = [], customFunc2 = [])
    
    dRange = intRange[2]-intRange[1]
    tmp = 0.
    no_table = false
    table = []
    f = []
    if func1=="tanh" && func2=="tanh"
        f1 = x->tanh.(x)
        f2 = x->tanh.(x)
        table = TABLE_TANH
    elseif func1=="Ptanh" && func2=="Ptanh"
        f1 = x->log.(cosh.(x))
        f2 = x->log.(cosh.(x))
        table = TABLE_PTANH
    elseif func1=="tanhP" && func2=="tanhP"
        f1 = x->1 ./((cosh(x)).^2)
        f2 = x->1 ./((cosh(x)).^2)
        table = TABLE_TANHP
    elseif func1=="tanhPP" && func2=="tanhPP"
        f1 = x->-2*tanh(x).*1 ./((cosh(x)).^2)
        f2 = x->-2*tanh(x).*1 ./((cosh(x)).^2)
        table = TABLE_TANHPP
    elseif func1=="tanh" && func2=="tanhPP"
        f1 = x->tanh.(x)
        f2 = x->-2*tanh(x).*1 ./((cosh(x)).^2)
        table = TABLE_TANH_TANHPP
    else
        f1 = customFunc1
        f2 = customFunc2
        method = "direct"
    end
    
    
    if delta0 == 0.
        tmp = 0.
    else
        if method=="table"
            try
                tmp = readfromtable(delta, delta0, DELTA_RANGE, DELTA0_RANGE, table)
            catch
                no_table = true
            end
            if tmp == NaN && correctNaN == true
                tmp = readfromtable(delta0, delta0, table)
            end
        end
        if method=="direct" || no_table == true
            for z in intRange
                tmp += 1 ./(2*pi) .* dRange .* exp.(-0.5 .*z.^2) .* 
                        sum(dRange .* exp.(-0.5 .*intRange.^2) .* f1.(sqrt(max(0.,delta0-(delta^2)/delta0)).*intRange + 
                        delta/sqrt(delta0) .* z) .* f2.(sqrt(delta0).*z))
            end
            #tmp += 1 ./(2*pi) .* dRange.*exp(-0.5 .*(z.^2)).*
            #    (sum(dRange.*exp(-0.5 .*(intRange.^2)).*Phi(sqrt(delta0-abs(delta)).*intRange +
            #    sqrt(abs(delta)).*z; J=J))).^2
        end
    end
    
    return tmp
    
end
export fAv2

function rhsstandardtheory(vars::Array{Float64,1}; tInd=1, delta0=1., intRange = -10.:0.1:10., g=0.5, func = "tanh" , method="table", correctNaN=false)

    out =  zeros(size(vars))
    out[1] = vars[2]
    out[2] = vars[1] - g^2 * fAv(vars[1],delta0, func; method=method, correctNaN=correctNaN, intRange=intRange, customFunc=func)
    return out

end
export rhsstandardtheory

function rhsAdaptTheory(vars::Array{Float64,1}; tInd=1, delta0=1., intRange = -10.:0.1:10., g=0.5, beta=1., gamma=1., 
                        func="tanh", funcP="tanhP", funcPP="tanhPP", Pfunc="Ptanh", method="table", correctNaN="correctNaN")
    out = zeros(size(vars))
    out[1] = vars[2]
    out[2] = vars[3]
    out[3] = vars[4]
    out[4] = (1+gamma^2-2*beta*gamma).*vars[3] - gamma^2 *(1+beta)^2 .* vars[1] +
              gamma^2 * g^2 * fAv(vars[1],delta0, func;intRange=intRange, method=method, correctNaN=correctNaN, customFunc=func) -
              g^2 .* vars[3] .* fAv(vars[1],delta0, funcP;intRange=intRange, method=method, correctNaN=correctNaN, customFunc=funcP) -
              g^2 .* vars[2].^2 .* fAv(vars[1],delta0,funcPP;intRange=intRange, method=method, correctNaN=correctNaN, customFunc=funcPP)
    return out
end    
export rhsAdaptTheory

function getVstandard(delta; delta0=1., g=0.5, Pfunc="Ptanh", intRange = -10:0.1:10, method="table", correctNaN=false)
    -0.5 .*delta.^2 + g^2 .*AdaptiveNet.fAv(delta,delta0,Pfunc; method=method, correctNaN=correctNaN, intRange=intRange, customFunc=Pfunc)
end

function findInitCondStandard(g; delta0min=0.01, delta0max=10., maxIter=100, epsilon=1e-6 ,
    intRange = -10:0.1:10, method="table", Pfunc="Ptanh", correctNaN=false, fullOut=false )

    # bisection method to find the root of V(Delta0) - V(0) #
    delta0 = 0.5 * (delta0max + delta0min)
    iter=0
    difference = 1.
    

    while abs(difference)>epsilon && iter<maxIter
        # verify that opposit signs
        if iter ==0
            diffMax = getVstandard(0.; delta0=delta0max, g=g , method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange) -
                getVstandard(delta0max; delta0=delta0max, g=g, method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange)
            diffMin = getVstandard(0.; delta0=delta0min, g=g, method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange) - 
                getVstandard(delta0min; delta0=delta0min, g=g, method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange)
            if sign(diffMax)==sign(diffMin)
                println("Change min and max please!")
                break
            end
        end
        
        iter += 1
        delta0 = 0.5 * (delta0max + delta0min)
        difference = getVstandard(0.; delta0=delta0, g=g , method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange) - 
            getVstandard(delta0; delta0=delta0, g=g, method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange)
        diffMax = getVstandard(0.; delta0=delta0max, g=g, method=method, Pfunc=Pfunc,correctNaN=correctNaN , intRange=intRange) - 
            getVstandard(delta0max; delta0=delta0max, g=g, method=method, Pfunc=Pfunc, correctNaN=correctNaN, intRange=intRange)
        if sign(difference) == sign(diffMax)
            delta0max = copy(delta0)
        elseif difference==0
            println("solution was found")
            break
        else
            delta0min = copy(delta0)
        end
        
    end
    if fullOut
        if abs(difference)<=epsilon
            return delta0, true, iter
        else
            return delta0, false, iter
        end
    else
        return delta0
    end
end
export findInitCondStandard

"
Returns the value of a quantity that might be putatively zero for the adaptive case
"
function getCondAdaptive(delta0; g=0.5, b=0.5, gamma=0.5, Pfunc="Ptanh", intRange = -10:0.1:10, method="table", correctNaN=false)

    delta0^2 ./ fAv(delta0,delta0, Pfunc ; intRange=intRange, method=method, correctNaN=correctNaN) - 2*g^2 ./ ((b+1)^2)

end

"
This is purely hypothetical.
"
function findInitCondAdaptive(g, b, gamma; delta0min=0.01, delta0max=10., maxIter=100, epsilon=1e-6, method="table", correctNaN=false )
    # bisection method to find the root of V(Delta0) - V(0) #
    delta0 = 0.5 * (delta0max + delta0min)
    maxIter = 200
    epsilon = 1e-6
    iter=0
    difference = 1.

    while abs(difference)>epsilon && iter<maxIter
        # verify that opposit signs
        if iter ==0
            diffMax = getCondAdaptive(delta0max; g=g, b=b, gamma=gamma, method=method, correctNaN=correctNaN)
            diffMin = getCondAdaptive(delta0min; g=g, b=b, gamma=gamma, method=method, correctNaN=correctNaN)
            if sign(diffMax)==sign(diffMin)
                println("Change min and max please!")
                break
            end
        end
        
        iter += 1
        delta0 = 0.5 * (delta0max + delta0min)
        difference = getCondAdaptive(delta0; g=g, b=b, gamma=gamma, method=method, correctNaN=correctNaN)
        diffMax = getCondAdaptive(delta0max; g=g, b=b, gamma=gamma, method=method, correctNaN=correctNaN)
        if sign(difference) == sign(diffMax)
            delta0max = copy(delta0)
        elseif difference==0
            println("solution was found")
            break
        else
            delta0min = copy(delta0)
        end
        
    end
    return delta0
end
export findInitCondAdaptive


function readfromtable(delta, delta0, deltaRange, delta0Range, table)
    i = findmin(abs.(delta - deltaRange))[2]
    j = findmin(abs.(delta0 - delta0Range))[2]
    return table[i,j]
end