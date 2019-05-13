### put together data from different table sections ###
using JLD

deltaRes = 0.001
tableType = "TANH"
tables = Array{Any}(5,9)
tableMerged = []
deltaRange = -5.:deltaRes:5.
delta0Range = 0.:deltaRes:5.

dataDir = "/home/samuel/mountpoint/"
for (j,tableType) in enumerate(["PTANH","TANH","TANHP","TANHPP","TANH_TANHPP"])
	for i=1:9
		filename = filter(x->(contains(x,string("table_",tableType)) && 
				   contains(x,string("deltaRes",string(deltaRes))) &&
				   contains(x,string("section",string(i)))),
			readdir(dataDir))[1]
		fi = jldopen(string(dataDir,filename),"r")
		tables[j,i] = read(fi,string("table",tableType))
		#deltaRange = read(fi,"deltaRange")
		#delta0Range = read(fi,"delta0Range")
		close(fi)
		if i==1
			tableMerged = copy(tables[j,i])
		else
			tableMerged = hcat(tableMerged,tables[j,i])
		end
	end
	fi = jldopen(string(dataDir,"table",tableType,
		"_deltaRes",string(deltaRes),"_merged.jld"),"w")
	fi[string("table",tableType)] = tableMerged
	fi["deltaRange"] = deltaRange
	fi["delta0Range"] = delta0Range
	close(fi)
end