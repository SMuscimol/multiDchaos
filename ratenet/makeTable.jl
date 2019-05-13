tableType = ARGS[1]
tableSection = parse(Int64, ARGS[2])

using ChaoticNet
using JLD

deltaRes = 0.1

dataDir = "/root/chaoticnet/data/"

#tablePTANH = []
#tableTANH = []
#tableTANHP = []
#tableTANHPP = []

#try
#	fi = jldopen(filename,"r")
#	try
#		tablePTANH = read(fi,"tablePTANH")
#	end
#	try
#		tableTANH = read(fi,"tableTANH")
#	end
#	try
#		tableTANHP = read(fi,"tableTANHP")
#	end
#	try
#		tableTANHPP = read(fi,"tableTANHPP")
#	end
#	close(fi)
#end
	
delta0RangeTot = 0.0:deltaRes:5.
deltaRange = -5.:deltaRes:5.
deltaHalfRange = 0.:deltaRes:5.

Ldelta0 = size(delta0RangeTot,1)
Lend = round(Int,Ldelta0./sqrt(10) .*sqrt(tableSection))
if tableSection==1
	Lstart = 1
else
	Lstart = round(Int,Ldelta0./sqrt(10) .*sqrt(tableSection-1) + 1 )
end
delta0Range = delta0RangeTot[Lstart:Lend]


if tableType=="tanh"
	Phi(x) = tanh.(x)
	filename = string(dataDir,"table_TANH_deltaRes",string(deltaRes),"_section",string(tableSection),".jld")
elseif tableType=="Ptanh"
	Phi(x) = log.(cosh.(x))
	filename = string(dataDir,"table_PTANH_deltaRes",string(deltaRes),"_section",string(tableSection),".jld")
elseif tableType=="tanhP"
	Phi(x) = 1./((cosh.(x)).^2)
	filename = string(dataDir,"table_TANHP_deltaRes",string(deltaRes),"_section",string(tableSection),".jld")
elseif tableType=="tanhPP"
	Phi(x) = -2.*tanh.(x).*1./((cosh.(x)).^2)
	filename = string(dataDir,"table_TANHPP_deltaRes",string(deltaRes),"_section",string(tableSection),".jld")
elseif tableType=="tanh-tanhPP"
	phi1(x) = tanh.(x)	 
	phi2(x) = -2.*tanh.(x).*1./((cosh.(x)).^2)
	filename = string(dataDir,"table_TANH_TANHPP_deltaRes",string(deltaRes),"_section",string(tableSection),".jld")
end

intRange = -10.:0.1:10.


table = zeros(size(deltaRange,1), size(delta0Range,1))
ind0 = findmin(abs.(deltaRange))[2]


for (j,delta0) in enumerate(delta0Range)
    for (i,delta) in enumerate(deltaHalfRange)
	    if abs(delta)>delta0
	       	table[ind0+i-1,j] = NaN
	       	table[ind0-i+1,j] = NaN
	    else
			if tableType=="tanh-tanhPP"
	       		table[ind0+i-1,j] = ChaoticNet.fAv2(delta,delta0,phi1,phi2;
					method="direct", customFunc1 = phi1, 
					customFunc2 = phi2, intRange = intRange)
	       		table[ind0-i+1,j] = copy(table[ind0+i-1,j])
			else
				table[ind0+i-1,j] = ChaoticNet.fAv(delta,delta0,Phi;
					method="direct", customFunc = Phi, intRange = intRange)
				table[ind0-i+1,j] = copy(table[ind0+i-1,j])
			end
       	end
    end
end

fi = jldopen(filename,"w")

if tableType=="tanh"
	tableTANH = copy(table)
	fi["tableTANH"] = tableTANH
elseif tableType=="Ptanh"
	tablePTANH = copy(table)
	fi["tablePTANH"] = tablePTANH
elseif tableType=="tanhP"
	tableTANHP = copy(table)
	fi["tableTANHP"] = tableTANHP
elseif tableType=="tanhPP"
	tableTANHP = copy(table)
	fi["tableTANHPP"] = copy(table)
elseif tableType=="tanh-tanhPP"
	tableTANH_TANHPP = copy(table)
	fi["tableTANH_TANHPP"] = copy(table)
end

fi["deltaRange"] = deltaRange
fi["delta0Range"] = delta0Range
fi["intRange"] = intRange
close(fi)
