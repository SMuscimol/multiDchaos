using StatsBase

function getMeanAutocorr(x::Array{Float64,2}; dLag = 1, initDiscard=1000)
    x = x[:,initDiscard:end]
    N = size(x,1)
    maxLag = round(Int,0.5*(size(x,2)-initDiscard)) # CHECK! I set the max lag to be half the length of the measurement - the discarded part
    if maxLag<=0
        println("ERROR, the measurement is too short.")
        return
    end
    lags = 0:dLag:maxLag #lags for the autocorrelation - in unit of the dt of the measurement!
    C = var(x) .* 1 ./N *sum(transpose(autocor(transpose(x), lags; demean=false)),1);
    return C[:], lags
end

function getMeanAutocorr(x::Array{Float64,2}, lags::StepRange{Int64,Int64}; initDiscard=1000)
    x = x[:,initDiscard:end]
    N = size(x,1)
    C = var(x) .* 1 ./N *sum(transpose(autocor(transpose(x), lags; demean=false)),1);
    return C[:]
end

export getMeanAutocorr

function findRelMaxima(f::Array{Float64,1}; epsilon=1e-4)
    # compute derivatives
    tmp = diff(f)
    df = vcat(tmp[1],tmp[:])
    tmp = diff(df)
    d2f = vcat(tmp[1],tmp[:])
    # compute zeros of first derivative
    zerosD = (df.>(zeros(size(df))-epsilon)).*(df.<(zeros(size(df))+epsilon))
    # compute maxima
    maxima = zerosD.*(d2f[:].<0.)
    maxIndices = find(x->x,maxima)
    # delete doubles
    diffMax = diff(maxIndices)
    toDelete = []
    for i in 2:size(maxIndices,1)
        if diffMax[i-1]==1
            toDelete = vcat(toDelete,i-1)
        end
    end
    deleteat!(maxIndices,toDelete)
    return maxIndices
end

export findRelMaxima