function getfilename(header::String, varValues::Dict{String,<:Any}, randomSignature="",extension=".jld")
	filename = header
	for (key,value) in varValues
		filename = string(filename, string("_",key,string(value)))
	end
	filename = string(filename,"_",randomSignature,extension)
end
export getfilename