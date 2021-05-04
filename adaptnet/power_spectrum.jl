"
Piecewise linear approximation of the hyperbolic tangent.
"
function piecewiseLinearGain(x;a=1.)
    -1 .*(x.<-a) + (x.<=a).*(x.>=-a).*x/(a) + 1 .*(x.>a)
end
export piecewiseLinearGain

"
Asymmetric generalization of the tanh. Used by Rajan et al., 2010
"
function asymmetrictanh(x; r0=0.1)
    (x<=0)*r0*tanh(x/r0) + (x>0)*(2-r0)*tanh(x/(2-r0))
end
export asymmetrictanh

"
Linear filter corresponding to the DMFT of the adaptive net.
"
function adaptFilter(freq; tauX=1., tauA=100.,beta=1.)
    omegas = 2 .* pi .* freq
    gamma = tauX./tauA
    if gamma>0
        return (1 .+ tauA^2 .* omegas.^2)./
            (tauX^2 .* tauA^2 .* omegas.^4 .+ (tauX^2 .+ tauA^2-2*beta*tauX*tauA).*omegas.^2 .+ (beta + 1)^2 )
    elseif gamma == 0
        return 1 ./
            (1 + tauX^2 .* omegas.^2 )
    end
end
export adaptFilter

function adaptfiltermulti(freq; gammas=[0.1], betas=[1.])
    omega = 2*pi*freq
    tmp = 1+im*omega + sum(gammas.*betas./(gammas + im*omega))
    return 1 ./((tmp).*conj(tmp))
end
export adaptfiltermulti

function adaptfilterhetero(freq; gamma=0.1, meanBeta=1., sigmaBeta=0.1)
    omega = 2*pi*freq
    chiSq = adaptFilter(freq; tauX=1., tauA=1 ./gamma, beta=meanBeta)
    ## IS THERE A PLUS OR A MINUS HERE!
    #return chiSq/(1+gamma^2*sigmaBeta^2*chiSq./(gamma^2+omega^2))
    return chiSq/(1-gamma^2*sigmaBeta^2*chiSq./(gamma^2+omega^2))
end
export adaptfilterhetero

function standardfilter(freq; tauX=1.)
    omegas = 2*pi .*freq
    1 ./(1 .+ tauX^2 .*omegas.^2)
end
export standardfilter

"
Given a power spectrum, it returns the effect of a piecewise linear approximation of the hyperbolic tangent on it,
Using an analytical formula I found.
"
function transformSpectrumPiecewiseLinear(Sx,a=1.,dInt=0.1;dt = 0.1, deltaT=10)

    Cxx = ifftshift((ifft(ifftshift(Sx)))).*1 ./(dt*deltaT) # looks like the inverse FT does not need a factor if the PS was properly
                                    # normalized. Or - it needs a factor 1 ./(dt*deltaT)
    #Cxx0 = Cxx[1]
    Sphi = 1 ./(a^2) .* (erf.(a./sqrt(2 .*Cxx[1]))).^2 .* Sx +
        4 ./(2 .*pi*a^2) .* exp(-a^2 ./ Cxx[1]) .*
        dt.*deltaT.*fftshift(fft(
            map(Cxxt->dInt.*sum(map(z->dInt.*sum(map(y->Cxx[1]./(2 .*sqrt.(1-y.^2)) .* (exp.(-a^2*y.^2 ./
            (Cxx[1]*(1-y.^2))).*min(realmax(Float64),sinh.(a^2 .* y./((1-y.^2).*Cxx[1]))) ),
                0.:dInt:z  ) ), 0.:dInt:max(0.,Cxxt-dInt))) , Cxx[:]./Cxx[1])
        ))
end
export transformSpectrumPiecewiseLinear



# function gF(x;a=1.)
#     -1 .*(x.<-a) + (x.<=a).*(x.>=-a).*x/(a) + 1 .*(x.>a)
# end

"
Given the power spectrum S, it samples a the Fourier transform of a signal that would result in the same power spectrum.
It takes S with the negative frequencies before.
"
function sampleFTfromspectrum(T, S; method="realImag", shift=true)
    if method=="polar"
        ### This is now done by sampling the phase from a uniform distribution and the amplitude from a Gaussian with
        ### zero mean and variance equal to the square root of S, time a sqrt of T for normalization.
        phases1Half = im.*2 .*pi.*rand(round(Int,0.5*size(S,1)))
        amp1half = sqrt(T).*sqrt.(S[1:round(Int,0.5*size(S,1))]).*
            randn(round(Int,0.5*size(S,1)))
        ### I concatenate the two halves + the zero frequency term, which for some reason I choose to have phase 0,
        ### probably for symmetry reasons.
        FT = vcat(amp1half,[sqrt(T).*sqrt.(S[round(Int,0.5*size(S,1))+1]).*randn()],
            amp1half[end:-1:1]).*exp.(vcat(phases1Half,0.,-phases1Half[end:-1:1]))
    elseif method=="realImag"
        ### The alternative option, maybe more correct, is to sample the real and imaginary parts, instead of the
        ### amplitude and phase.
        S = max.(zeros(size(S)),real.(S))
        HS = round(Int,0.5 .*size(S,1)) ## this should return the 0-freq index?
        if shift
            # realPart = sqrt.(T.*0.5 .*S[2:HS]).*randn(HS-1)
            # imagPart = sqrt.(T.*0.5 .*S[2:HS]).*randn(HS-1)
            #FT = vcat(sqrt.(T.*S[1]).*randn(), realPart+im.*imagPart, sqrt.(T.*S[HS+1]).*randn(), (realPart-im.*imagPart)[end:-1:1])
            realPart = sqrt.(T.*0.5 .*S[1:HS-1]).*randn(HS-1)
            imagPart = sqrt.(T.*0.5 .*S[1:HS-1]).*randn(HS-1)
            FT = vcat(realPart+im.*imagPart, sqrt.(T.*S[HS]).*randn(), (realPart-im.*imagPart)[end:-1:1], sqrt.(T.*S[end]).*randn())
            #FT = vcat(realPart+im.*imagPart, sqrt.(0.5 .*T.*S[HS]).*randn(), (realPart-im.*imagPart)[end:-1:1], sqrt.(0.5 .*T.*S[end]).*randn())

        else
            realPart = sqrt.(T.*0.5 .*S[2:HS]).*randn(HS-1)
            imagPart = sqrt.(T.*0.5 .*S[2:HS]).*randn(HS-1)
            FT = vcat( sqrt.(T.*S[1]).*randn(), realPart+im.*imagPart, sqrt.(T.*S[HS+1]).*randn(), (realPart -im.*imagPart)[end:-1:1] )
        end
    end
    return FT

end

"
Test with spectrum from -maxFreq:dFreq:maxFreq-dFreq
"
function sampleFTfromspectrum2(freqRange, S; method="realImag", shift=true)
    if method=="polar"
        ### This is now done by sampling the phase from a uniform distribution and the amplitude from a Gaussian with
        ### zero mean and variance equal to the square root of S, time a sqrt of T for normalization.
        phases1Half = im.*2 .*pi.*rand(round(Int,0.5*size(S,1)))
        amp1half = sqrt(tMax).*sqrt.(S[1:round(Int,0.5*size(S,1))]).*
            randn(round(Int,0.5*size(S,1)))
        ### I concatenate the two halves + the zero frequency term, which for some reason I choose to have phase 0,
        ### probably for symmetry reasons.
        FT = vcat(amp1half,[sqrt(tMax).*sqrt.(S[round(Int,0.5*size(S,1))+1]).*randn()],
            amp1half[end:-1:1]).*exp.(vcat(phases1Half,0.,-phases1Half[end:-1:1]))
    elseif method=="realImag"
        #tMax = 0.5*(size(freqRange,1)-1)
        maxFreq = freqRange[end]
        dFreq = freqRange[2]-freqRange[1]
        #tMax = 2 .*maxFreq./dFreq ### 1 ./dt * t_max
        tMax = (((size(freqRange,1).*dFreq))).*(size(freqRange,1)-1) ### 1 ./dt * t_max -- to understand why the factor in front
        HS = find(x->x==0, freqRange)[1]
        ### tMaxhe alternative option, maybe more correct, is to sample the real and imaginary parts, instead of the
        ### amplitude and phase.
        S = max.(zeros(size(S)),real.(S))
        if shift
            # realPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
            # imagPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
            #FT = vcat(sqrt.(tMax.*S[1]).*randn(), realPart+im.*imagPart, sqrt.(tMax.*S[HS+1]).*randn(), (realPart-im.*imagPart)[end:-1:1])
            realPart = sqrt.(tMax.*0.5 .*S[1:HS-1]).*randn(HS-1)
            imagPart = sqrt.(tMax.*0.5 .*S[1:HS-1]).*randn(HS-1)
            FT = vcat(realPart+im.*imagPart, sqrt.(tMax.*S[HS]).*randn(), (realPart-im.*imagPart)[end:-1:1])
            #FT = vcat(realPart+im.*imagPart, sqrt.(0.5 .*tMax.*S[HS]).*randn(), (realPart-im.*imagPart)[end:-1:1], sqrt.(0.5 .*tMax.*S[end]).*randn())

        else
            realPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
            imagPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
            FT = vcat( sqrt.(tMax.*S[1]).*randn(), realPart+im.*imagPart, sqrt.(tMax.*S[HS+1]).*randn(), (realPart -im.*imagPart)[end:-1:1] )
        end
    end
    return FT

end

"
Given a power spectrum, it returns the numerically evaluated
effect of a nonlinear function gF on it.
"
function numericalTransformSpectrum(freqRange, dt, T, M, S, gF::Function; method="realImag", shift=true, center=false, substractMean=true)

    #FTx = Array{Complex{Float64},2}(zeros(M,size(freqRange,1)))
    #FTx = Array{Complex{Float64},2}(zeros(size(freqRange,1)))
    SxPhi = zeros(size(S))
    meanPhi = 0.
    dFreq = freqRange[2] - freqRange[1]
    for i =1:M
        #FTx[i,:] = sampleFTfromspectrum(T,S; method=method, shift=shift)
        # FTx = sampleFTfromspectrum(T,S; method=method, shift=shift)
        FTx = sampleFTfromspectrum2(freqRange,S; method=method, shift=shift)

        if shift
            x = real.(ifftshift(ifft(ifftshift(FTx))))
            #apply gain
            phiX = gF.(x)
            if center
                phiX = phiX - mean(phiX)
            end
            #transform back
            phiFT = fftshift(fft(phiX))
        else
            x = real.(ifft(FTx,2))
            phiX = gF.(x)
            if center
                phiX = phiX - mean(phiX)
            end
            phiFT = fftshift(fft(phiX))
        end
        # average to get the spectrum of phi

        meanPhi = (i-1)./i .* meanPhi + 1 ./i .* mean(phiX)
        #SxPhi = (i-1)./i .* SxPhi +
        #    1 ./i .*2 ./(T) .*conj.(phiFT).*phiFT ### version with the factor 2
        # SxPhi = (i-1)./i .* SxPhi +
        #     1 ./i .*1 ./(T) .*conj.(phiFT).*phiFT ### version without the factor 2
        SxPhi = (i-1)./i .* SxPhi +
             1 ./i .*1 ./((((size(freqRange,1).*dFreq))).*T) .*conj.(phiFT).*phiFT ### version without the factor 2
    end
    if substractMean
        SxPhi[find(x->x==0,freqRange)] += -(meanPhi^2)./(dFreq)
    end
    return SxPhi
end

function numericalnonlinearpass(S, gF::Function; dFreq=0.0002, maxFreq=1., M=100, method="realImag", withCosDrive=false, cosAmp=[],cosFreq=[],
     shift=true, center=false, substractMean=true)


    if shift
        #freqRange = -(maxFreq-dFreq):dFreq:maxFreq
        freqRange = -maxFreq:dFreq:maxFreq
    else
        freqRange = vcat( 0.:dFreq:maxFreq, -(maxFreq-dFreq):dFreq:-dFreq)
    end
    tMax = 1 ./dFreq
    dt = 1 ./(2 .*maxFreq)
    T = tMax./dt
    tRange = gettrangecorr(freqRange)
    #FTx = Array{Complex{Float64},2}(zeros(M,size(freqRange,1)))
    #FTx = Array{Complex{Float64},2}(zeros(size(freqRange,1)))
    SxPhi = zeros(size(S))
    meanPhi = 0.

    for i =1:M
        #FTx[i,:] = sampleFTfromspectrum(T,S; method=method, shift=shift)
        # FTx = sampleFTfromspectrum(T,S; method=method, shift=shift)
        FTx = sampleFTfromspectrum2(freqRange,S; method=method, shift=shift)


        if shift
            x = real.(ifftshift(ifft(ifftshift(FTx))))
            if withCosDrive
                x = x + cosAmp.*cos.(2*pi*cosFreq.*tRange + 2*pi*rand())
            end
            #apply gain
            phiX = gF.(x)
            if center
                phiX = phiX - mean(phiX)
            end
            #transform back
            phiFT = fftshift(fft(phiX))
        else
            x =real.(ifft(FTx,2))
            if withCosDrive
                x = x + cosAmp.*cos.(2*pi*cosFreq.*tRange + 2*pi*rand())
            end
            phiX = gF.(x)
            if center
                phiX = phiX - mean(phiX)
            end
            phiFT = fftshift(fft(phiX))
        end
        # average to get the spectrum of phi

        meanPhi = (i-1)./i .* meanPhi + 1 ./i .* mean(phiX)
        #SxPhi = (i-1)./i .* SxPhi +
        #    1 ./i .*2 ./(T) .*conj.(phiFT).*phiFT ### version with the factor 2
        # SxPhi = (i-1)./i .* SxPhi +
        #     1 ./i .*1 ./(T) .*conj.(phiFT).*phiFT ### version without the factor 2
        SxPhi = (i-1)./i .* SxPhi +
             1 ./i .*1 ./((((size(freqRange,1).*dFreq))).*T) .*conj.(phiFT).*phiFT ### version without the factor , but with a weird factor in front
    end
    if substractMean
        SxPhi[find(x->x==0,freqRange)] += -(meanPhi^2)./(dFreq)
    end
    return SxPhi
end

function cubicnonlinearpass(S, gF::Function; dFreq=0.001, maxFreq=0.5)

    nonlinearpassclosedform(S,gF; dFreq=dFreq, maxFreq=maxFreq, closedform = (Cx,Cx0)->((1+(Cx0)^2 -2*Cx0).*Cx + 2 ./3 .*Cx.^3 ) )

end

"
New version of the calculation coming from Price's theorem
"
function PWLnonlinearpass(S, gF::Function; dFreq=0.001, maxFreq=0.5, dIntSigmaFactor = 100)

    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* real.(ifft(ifftshift(S)))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0]
    dIntSigma = cX0 ./dIntSigmaFactor;
    cPhi = zeros(size(cX))
    #tmpFun = s -> 1 ./sqrt(1-s^2 ./(cX0^2)) * exp(-1 ./(cX0*(1-s^2 ./(cX0.^2) ))) * sinh(s./(cX0^2*(1-s^2 ./(cX0^2) )))
    tmpFun = s-> 1 ./2 .* 1 ./sqrt(1-s^2 ./(cX0^2)) * (
                    exp(-(1 .-s./cX0)./(cX0*(1-s^2 ./(cX0.^2) ))) -
                    exp(-(1 .+s./cX0)./(cX0*(1-s^2 ./(cX0.^2) ))) )
    for i=1:size(cX,1)
        if cX[i]>=0
            cPhi[i] = erf(1 ./sqrt(2*cX0))^2 * cX[i] + 2 ./(pi*cX0).* (dIntSigma).^2 .*
            sum(map( sigmaP->sum( tmpFun.(0.:dIntSigma:sigmaP) ),
                    0:dIntSigma:max(cX[i]-dIntSigma,dIntSigma) ))
        # elseif cX[i]<0
        #     cPhi[i] = erf(1 ./sqrt(2*cX0))^2 * cX[i] - 2 ./(pi*cX0).* (dIntSigma).^2 .*
        #     sum(map( sigmaP->sum( tmpFun.(0.:dIntSigma:sigmaP) ),
        #             min(cX[i]+dIntSigma,-dIntSigma):dIntSigma:0.))
        elseif cX[i]<0
            cPhi[i] = erf(1 ./sqrt(2*cX0))^2 * cX[i] - 2 ./(pi*cX0).* (dIntSigma).^2 .*
            sum(map( sigmaP->sum( tmpFun.(0.:dIntSigma:sigmaP) ),
                    0:dIntSigma:max(-cX[i]-dIntSigma,dIntSigma) ))
        end
    end
    #cReconstructed = copy(cX);
    #cReconstructed[round(Int,0.5*size(cX,1))+1:end] = cReconstructed[round(Int,0.5*size(cX,1))+1:-1:1]
    sPhi = dt.*fftshift(fft(cPhi));

end
export PWLnonlinearpass

"
Alternative implementation - requires QuadGK to be installed
"
function PWLnonlinearpass2(S, gF::Function; dFreq=0.001, maxFreq=0.5, dIntSigmaFactor = 100)

    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* real.(ifft(ifftshift(S)))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0]
    dIntSigma = cX0 ./dIntSigmaFactor;
    cPhi = zeros(size(cX))
    tmpFun = s-> 1 ./2 .* 1 ./sqrt(1-s^2 ./(cX0^2)) * (
                    exp(-(1 .-s./cX0)./(cX0*(1-s^2 ./(cX0.^2) ))) -
                    exp(-(1 .+s./cX0)./(cX0*(1-s^2 ./(cX0.^2) ))) )
    for i=1:size(cX,1)
        if cX[i]>=0
            cPhi[i] = erf(1 ./sqrt(2*cX0))^2 * cX[i] + 2 ./(pi*cX0).* (dIntSigma).^2 .*
            sum(map( sigmaP-> quadgk(tmpFun, 0., sigmaP)[1],
                    0:dIntSigma:max(cX[i]-dIntSigma,dIntSigma) ))
        elseif cX[i]<0
            cPhi[i] = erf(1 ./sqrt(2*cX0))^2 * cX[i] - 2 ./(pi*cX0).* (dIntSigma).^2 .*
            sum(map( sigmaP-> quadgk(tmpFun, 0., sigmaP)[1],
                    0:dIntSigma:max(-cX[i]-dIntSigma,dIntSigma) ))
        end
    end

    sPhi = dt.*fftshift(fft(cPhi));

end
export PWLnonlinearpass2

function PWLnonlinearpassseries(S, gF::Function; dFreq=0.001, maxFreq=0.5, dIntSigmaFactor=100, truncateAt=3, returnAllTerms=false)

    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* real.(ifft(ifftshift(S)))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0]
    cPhi = zeros(size(cX))
    aS, aSold, aSoldold = [],[],[]
    if returnAllTerms
        terms = Array{Any}(truncateAt)
    end

    for i=1:truncateAt
        if i==1
            sTerm = erf(1 ./sqrt(2*cX0))^2 .* cX
        elseif i>1
            aSoldold = copy(aSold)
            aSold = copy(aS)
            if i == 2
                aS = [1]
            elseif i==3
                aS = [0,1]
            elseif i>3
                aS = zeros(i-1)
                aSold = vcat(aSold, zeros(i-1-size(aSold,1)))
                aSoldold = vcat(aSoldold,zeros(i-1-size(aSoldold,1)))
                aS[1] = -(i-3)*aSoldold[1]
                for k=2:i-1
                    aS[k] = aSold[k-1] - (i-3)*aSoldold[k]
                end

            end
            if mod(i,2)==1
                sTerm = 2 ./(pi*cX0^(i-1)).*exp.(-1 ./cX0).*(transpose(aS) * ((1 ./sqrt(cX0)).^(0.:1.:(i-2))))^2 .* cX.^(i)
                sTerm = 1 ./(factorial(i)).*sTerm[:]
            else
                sTerm = 0
            end


        end

        if returnAllTerms
            terms[i] = copy(sTerm)
        end

        cPhi[:] = cPhi[:] + sTerm
    end
    if returnAllTerms
        dt.*fftshift(fft(cPhi)), terms
    else
        return dt.*fftshift(fft(cPhi))
    end
end
export PWLnonlinearpassseries

function PWLpass_1(S, gF::Function; dFreq=0.001, maxFreq=0.5, dIntSigmaFactor = 100)
    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* real.(ifft(ifftshift(S)))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0]

    cPhi = erf(1 ./sqrt(2*cX0)).^2 .* cX

    return dt .* fftshift(fft(cPhi));
end

function PWLpass_1_2(S, gF::Function; dFreq=0.001, maxFreq=0.5, dIntSigmaFactor = 100)
    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* real.(ifft(ifftshift(S)))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0]

    cPhi = erf(1 ./sqrt(2*cX0)).^2 .* cX +
        1 ./pi .* 1 ./(cX0^3) .* exp(-1 ./cX0) .*(cX.^3)

    return dt .* fftshift(fft(cPhi));
end
function PWLpass_1_3(S, gF::Function; dFreq=0.001, maxFreq=0.5, dIntSigmaFactor = 100)
    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* real.(ifft(ifftshift(S)))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0]

    cPhi = erf(1 ./sqrt(2*cX0)) .^2 .* cX +
        1 ./pi .* 1 ./(cX0^3) .* exp(-1 ./cX0) .*(cX .^3) +
        1 ./(3 .*pi) .* 1 ./(cX0^4) .* exp(-1 ./cX0) .* ((1 ./sqrt(cX0))^3-3*1 ./sqrt(cX0))^2 .* cX.^5

    return dt.*fftshift(fft(cPhi));
end
export PWLpass_1, PWLpass_1_2, PWLpass_1_3



function nonlinearpassclosedform(S, gF::Function; dFreq=0.001, maxFreq=0.5, closedform = (Cx,Cx0)->Cx )

    freqRange = -maxFreq:dFreq:maxFreq
    dt = 1 ./(2 .*maxFreq)
    cX = dFreq .* size(S,1) .* ifft(ifftshift(S))
    index0 = 1 #the index that indicate the autocorrelation at 0-time lag
    cPhi = closedform(cX,cX[index0])
    return dt.*fftshift(fft(cPhi))

end

function convolutionpass(S, gF::Function; convolutionFunction = [] )
    (conv(S, convolutionFuntion))[1:size(S,1)]
end

"
Get distance between two spectra, after having applied some smoothing with smoothing scale.
"
function getspectradistance(spectrum1, spectrum2; smoothingScale=10)

    # normalize #
    sp1 = spectrum1 ./(maximum(abs.(spectrum1)))
    sp2 = spectrum2 ./(maximum(abs.(spectrum1)))
    # def filter for smoothing and smooth
    filt = exp.(-1 ./(smoothingScale^2) .* (-50:1:50).^2)
    spF1 = conv( filt , real.(sp1))
    spF2 = conv( filt, real.(sp2))

    return 2 ./(sum(spF1 + spF2)) .* sum(abs.(spF1-spF2))

end

"
General way to compute the nonlinear pass based on the closed form expression in time domain (two Gaussian integration needed)
"
function generalnonlinearpass(S, gF::Function;
    dFreq=0.001, maxFreq=0.5, externalDrive=false, externalSolution=[], externalParamPDF=[], externalParamRange=[], intRange=-5:0.1:5)

    freqRange = -maxFreq:dFreq:maxFreq
    tRange = gettrangecorr(freqRange)
    cX = getautocorrfromspectrum(freqRange, S)
    cX = real.(cX)
    dt = tRange[2]-tRange[1]
    index0 = findmin(abs.(tRange))[2] + 1
    cX0 = cX[index0]
    dRange = intRange[2]-intRange[1]
    cPhi = zeros(size(cX))
    if externalDrive
        dExtParam = externalParamRange[2]-externalParamRange[1]
    end

    for i=1:size(cX,1)
        if externalDrive
            cPhi[i] =  1 ./(2*pi) .* dRange^2 .* sum(map(z->exp.(-0.5 .*z.^2).*sum( exp.(-0.5 .*intRange.^2) .*
                dExtParam .* sum(extParam->externalParamPDF[j].*
                    gF.(sqrt(max(0.,cX0-(cX[i]^2)/cX0)).*intRange + cX[i]/sqrt(cX0) .* z + externalSolution(tRange[i],extParam)) .*
                    gF.(sqrt(cX0).*z + externalSolution[i,j])),externalParamRange), intRange ) )
        else
            cPhi[i] = 1 ./(2*pi) .* dRange^2 .* sum(map(z->exp.(-0.5 .*z.^2).*sum( exp.(-0.5 .*intRange.^2) .*
                gF.(sqrt(max(0.,cX0-(cX[i]^2)/cX0)).*intRange + cX[i]/sqrt(cX0) .* z ) .*
                gF.(sqrt(cX0).*z ) ), intRange ) )
        end
    end

    sPhi = dt.*fftshift(fft(fftshift(cPhi)));

end

export generalnonlinearpass



"
Given a starting power spectrum S0, it iteratively applies the linear filter of the adaptive net and then a static nonlinearity gF,
in order to find a solution of the self-consistent equation.
"
function numericalIterativeNetSpectrum(iterations, dFreq, maxFreq, M, S0, g,  gF::Function; netType="adapt", tauX=1., tauA=100., beta=1.,
                                       externalInput=false, externalSpectrum=[], method="realImag", shift=true,
                                       stopAtConvergence=false, distanceTh=1e-2, center=false, substractMean=true)

    #deltaFreq = freqRange[2]-freqRange[1]
    if shift
        #freqRange = -(maxFreq-dFreq):dFreq:maxFreq
        freqRange = -maxFreq:dFreq:maxFreq
    else
        freqRange = vcat( 0.:dFreq:maxFreq, -(maxFreq-dFreq):dFreq:-dFreq)
    end
    tMax = 1 ./dFreq
    dt = 1 ./(2 .*maxFreq)
    T = tMax./dt
    SxR = Array{Any}(iterations)
    SxR[1] = copy(S0)
    if netType=="adapt"
        F = adaptFilter(freqRange; tauX=tauX,tauA=tauA,beta=beta)
    elseif netType=="standard"
        F = standardfilter(freqRange; tauX=tauX)
    end

    for iter in 1:iterations-1
        # static nonlinear pass
        SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, SxR[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
        # scale with the gain and linear filtering
        if externalInput
            SxR[iter+1] = F .* (g^2 .* SxPhi + externalSpectrum )
        else
            SxR[iter+1] = F .* g^2 .* SxPhi
        end
        if stopAtConvergence && (getspectradistance(SxR[iter], SxR[iter+1]) < distanceTh)
            SxR = SxR[1:iter+1]
            break
        end
    end
    return SxR
end
export numericalIterativeNetSpectrum

function iterativemethod(iterations, freqRange, S0, g, gF::Function, nonlinearpass::Function; saveAll=true, netType="adapt", tauX=1., tauA=100., beta=1.,
                                       externalInput=false, externalSpectrum=[], withCosDrive=false, stopAtConvergence=false, distanceTh=1e-2, verbose=false,
                                       nlPassArgs... )
    if saveAll
        SxR = Array{Any}(undef, iterations)
        SxR[1] = copy(S0)
        if netType=="adapt"
            F = adaptFilter(freqRange; tauX=tauX,tauA=tauA,beta=beta)
        elseif netType=="standard"
            F = standardfilter(freqRange; tauX=tauX)
        end

        for iter in 1:iterations-1
            if verbose
                println("iteration: ",iter)
            end
            # static nonlinear pass
            #SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, SxR[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
            if withCosDrive
                SxPhi = numericalnonlinearpass(SxR[iter], gF; withCosDrive=true, nlPassArgs...)
            else
                SxPhi = nonlinearpass(SxR[iter], gF; nlPassArgs...)
            end
            # scale with the gain and linear filtering
            if externalInput
                SxR[iter+1] = F .* (g^2 .* SxPhi + externalSpectrum )
            elseif externalInput==false
                SxR[iter+1] = F .* g^2 .* SxPhi
            end
            if stopAtConvergence && (getspectradistance(SxR[iter], SxR[iter+1]) < distanceTh)
                SxR = SxR[1:iter+1]
                break
            end
        end
        return SxR
    else

        Sx = copy(S0)
        if netType=="adapt"
            F = adaptFilter(freqRange; tauX=tauX,tauA=tauA,beta=beta)
        elseif netType=="standard"
            F = standardfilter(freqRange; tauX=tauX)
        end

        for iter in 1:iterations-1
            if verbose
                println("iteration: ",iter)
            end
            # static nonlinear pass
            #SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, SxR[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
            if withCosDrive
                SxPhi = numericalnonlinearpass(Sx, gF; withCosDrive=true, nlPassArgs...)
            else
                SxPhi = nonlinearpass(Sx, gF; nlPassArgs...)
            end
            # scale with the gain and linear filtering
            if externalInput
                Sx = F .* (g^2 .* SxPhi + externalSpectrum )
            else
                Sx = F .* g^2 .* SxPhi
            end
            ### TO DO: IMPLEMENT STOPPING CRITERION WHEN NOT SAVING ###
            # if stopAtConvergence && (getspectradistance(SxR[iter], SxR[iter+1]) < distanceTh)
            #     SxR = SxR[1:iter+1]
            #     break
            # end
        end
        return Sx
    end
end
export iterativemethod

function iterativemethodgeneral(iterations, freqRange, S0, g, gF::Function, nonlinearpass::Function, filt; saveAll=true, tauX=1., tauA=100., beta=1.,
                                       externalInput=false, externalSpectrum=[], stopAtConvergence=false, distanceTh=1e-2,
                                       nlPassArgs... )

    if saveAll
        SxR = Array{Any}(iterations)
        SxR[1] = copy(S0)


        for iter in 1:iterations-1
            # static nonlinear pass
            #SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, SxR[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
            SxPhi = nonlinearpass(SxR[iter], gF; nlPassArgs...)
            # scale with the gain and linear filtering
            if externalInput
                SxR[iter+1] = filt .* (g^2 .* SxPhi + externalSpectrum )
            else
                SxR[iter+1] = filt .* g^2 .* SxPhi
            end
            if stopAtConvergence && (getspectradistance(SxR[iter], SxR[iter+1]) < distanceTh)
                SxR = SxR[1:iter+1]
                break
            end
        end
        return SxR
    else

        Sx = copy(S0)


        for iter in 1:iterations-1
            # static nonlinear pass
            #SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, SxR[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
            SxPhi = nonlinearpass(Sx, gF; nlPassArgs...)
            # scale with the gain and linear filtering
            if externalInput
                Sx = filt .* (g^2 .* SxPhi + externalSpectrum )
            else
                Sx = filt .* g^2 .* SxPhi
            end
            # if stopAtConvergence && (getspectradistance(SxR[iter], SxR[iter+1]) < distanceTh)
            #     SxR = SxR[1:iter+1]
            #     break
            # end
        end
        return Sx
    end

end
export iterativemethodgeneral

"
Runs the iterative method but looking only for limit-cycle solutions, i.e. sums of delta peaks.
It assumes that there is a driving at the frequency fI and amplitude amp, and that the limit cycle can only have fundamental frequency fI.
"
function iterativelimitcycle(iterations, freqRange, S0, fI::Float64, g::Float64, gF::Function, filt, amp::Float64;
                                nMax=3, saveAll=true, nlPassArgs... )
    bs = Array{Any}(iterations)
    if saveAll
        Sx = Array{Any}(iterations)
        Sx[1] = copy(S0)
        for iter in 1:iterations-1
            # static nonlinear pass
            #SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, Sx[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
            bs[iter] = measureosccoefficients(freqRange, fI, Sx[iter]; nMax=nMax)
            SxPhi = numericalnonlinearpass(Sx[iter], gF; withCosDrive=true, cosAmp=amp, cosFreq=fI, nlPassArgs... )
            removenonosc!(SxPhi, freqRange, fI; nMax=nMax)
            # scale with the gain and linear filtering
            Sx[iter+1] = filt .* g^2 .* SxPhi
        end
        bs[end] = measureosccoefficients(freqRange, fI, Sx[end]; nMax=nMax)
        return Sx, bs
    else

        Sx = copy(S0)
        for iter in 1:iterations-1
            # static nonlinear pass
            #SxPhi = numericalTransformSpectrum(freqRange, dt, T, M, Sx[iter], gF; method=method, shift=shift, center=center, substractMean=substractMean)
            bs[iter] = measureosccoefficients(freqRange, fI, Sx; nMax=nMAx)
            SxPhi = numericalnonlinearpass(Sx, gF; withCosDrive=true, cosAmp=amp, cosFreq=fI, nlPassArgs... )
            removenonosc!(SxPhi, freqRange, fI; nMax=nMax)
            # scale with the gain and linear filtering
            Sx = filt .* g^2 .* SxPhi
        end
        bs[end] = measureosccoefficients(freqRange, fI, Sx; nMax=nMax)
        return Sx, bs
    end
end
export iterativelimitcycle

"
Extracts the amplitudes of the delta peaks of a PSD at a frequnecy fI and it multiples up to nMax*fI
"
function measureosccoefficients(freqRange, fI::Float64, S; nMax=3)
    map(n->S[find(x->abs(x-n*fI)<=eps(),freqRange)],1:nMax)
end

function removenonosc!(S, freqRange, fI::Float64; nMax=3)
    peakFreqs = fI:fI:nMax*fI
    S[find(x->all(abs.(abs(x)-peakFreqs).>eps()),freqRange)] = 0.
end

"
Power spectrum of a damped harmonic oscillator.
"
function appSpectrum(freq;params=[1 1 1])
    omega = 2*pi.*freq
    2 .*params[1]./((omega.^2 - params[2].^2).^2 + params[3].^2 .*omega.^2)
end

function findPeak(Sx, freqRange)
    zeroFreqInd = find(x->x==0,freqRange)[1]
    maxValue, maxInd = findmax(real.(Sx[zeroFreqInd:end]))
    return maxValue, zeroFreqInd + maxInd -1
end
#function costFunction(x)
#    sum((real(Sx) - appSpectrum(freqRange;params=x)).^2)
#end

function computeQfactor(Sx, freqRange; optimizer=NelderMead)
    out = optimize( x->sum(real(Sx) - appSpectrum(freqRange;params=x)).^2 , ones(3), optimizer())
    Q = sqrt(out.minimizer[2].^2 - out.minimizer[3].^2 ./4)./out.minimizer[3]
    return Q, out.minimizer, out.minimum, out.g_converged
end
export computeQfactor

function computeQfactorHeight(Sx, freqRange)
    ### Find the position of the resonance ###
    maxSValue, freqInd = findPeak(Sx, freqRange)
    ### Find width
    i = 0
    width = 0.
    height = copy(maxSValue)
    oldHeight = copy(height)
    while (height./maxSValue > 0.5) && ((freqInd+i)<size(Sx,1))
        i+=1
        oldHeight = copy(height)
        height = real(Sx[freqInd+i])
    end
    if i<size(Sx,1)-freqInd
        width =  2 .*i
    else
        println("The peak is not sharp enough")
        return
    end
    return freqRange[freqInd] ./ (freqRange[freqInd+i]-freqRange[freqInd-i])
end
export computeQfactorHeight

function findresfreq(gamma, beta; verbose=false)
    if gamma^2 > sqrt(beta*gamma^2*(beta + 2 +2*gamma))
        if verbose
            println("There is no resonance frequency for these parameters")
        end
        return NaN
    else
        return 1 ./(2*pi)*sqrt(-gamma^2 + sqrt(beta*gamma^2*(beta+2+2*gamma)) )
    end
end
export findresfreq

function findamp(gamma, beta)
    if gamma^2 > sqrt(beta*gamma^2*(beta + 2 +2*gamma))
        return adaptFilter(0.; tauX=1., tauA=1 ./gamma,beta=beta)
    else
        return adaptFilter(findresfreq(gamma,beta;verbose=false); tauX=1., tauA=1 ./gamma,beta=beta)
    end
end
export findamp

function findfreqwidthnumerical(gamma, b, freqRange;verbose=false)
    if gamma^2 > sqrt(b*gamma^2*(b + 2 +2*gamma))
        if verbose
            println("There is no resonance frequency for these parameters")
        end
        return NaN
    else
        chiSq = adaptFilter(halfFreqRange; tauX=1, tauA=1 ./gamma, beta=b)
        return 2 .*(halfFreqRange[findmin( abs.(chiSq - 0.5*maximum(chiSq)))[2]] - findresfreq(gamma,b))
    end
end
export findfreqwidthnumerical

function findfreqwidth(gamma, b;verbose=false)
    if gamma^2 > sqrt(b*gamma^2*(b + 2 +2*gamma))
        if verbose
            println("There is no resonance frequency for these parameters")
        end
        return NaN
    else
        k1 = sqrt(b*(2+b+2*gamma))
        return 1 ./(2*pi)*(-2*sqrt(gamma*(k1-gamma)) + sqrt(2)*sqrt(1+(4*k1-2*b-3*gamma)*gamma +
                sqrt(16*b^2*gamma^2-(1+8*k1*gamma-gamma^2)*(-1+gamma^2) +
                4*b*gamma*(-1+gamma*(6-4*k1+7*gamma) )  )) )
    end
end
export findfreqwidth

function findQfactor1(gamma, b; verbose=false)
    if gamma^2 > sqrt(b*gamma^2*(b + 2 +2*gamma))
        if verbose
            println("There is no resonance frequency for these parameters")
        end
        return NaN
    else
        return findresfreq(gamma, b; verbose=verbose)./findfreqwidth(gamma,b; verbose=verbose)
    end
end

function findQfactor1(gamma, b, freqRange; verbose=false)
    if gamma^2 > sqrt(b*gamma^2*(b + 2 +2*gamma))
        if verbose
            println("There is no resonance frequency for these parameters")
        end
        return NaN
    else
        return findresfreq(gamma, b; verbose=verbose)./
            findfreqwidthnumerical(gamma,b, freqRange; verbose=verbose)
    end
end

function getQfactorfromautocorrelation(tRange, C, fp::Float64)
    period = 1 ./fp
    t0Ind = findmin(abs.(tRange))[2]
    return 1 ./period .* tRange[findmin( abs.(abs.(C[t0Ind:end])- exp(-1) .* C[t0Ind]) )[2]]
end
export getQfactorfromautocorrelation

export findQfactor1

function smoothSpectrum(S ; smoothScale=10)
    filRange = -10*smoothScale:1:10*smoothScale
    fil = 1 ./(sqrt(2*pi*smoothScale^2)).*exp.(-1 ./(2*smoothScale^2).*(filRange).^2);
    return conv(fil, real.(S))[round(Int,0.5*size(filRange,1)):end-round(Int,0.5*size(filRange,1))-1]
end
export smoothSpectrum

"
Get resonance frequency from measured spectrum.
"
function getresfreq(freqRange, S)
    abs.(freqRange[findmax(real.(S))[2]])
end
export getresfreq

function getamp(S)
    findmax(real.(S))[1]
end
export getamp

"
Get the frequency width at half-max from measured spectrum.
"
function gethalfmaxdelta(freqRange, S)
    resFreq = getresfreq(freqRange, S)
    zeroFreqInd = findmin(abs.(freqRange))[2]
    resFreqInd = findmin(abs.(resFreq-freqRange))[2]
    2 .*abs.(resFreq - abs.(freqRange[(resFreqInd -1) + findmin(abs.(real.(S[resFreqInd:end])-0.5 .*S[resFreqInd]) )[2]]) )
end
export gethalfmaxdelta

"
Get Q-factor from measured spectrum.
"
function getQfactor(freqRange, S)
    getresfreq(freqRange,S)/gethalfmaxdelta(freqRange, S)
end
export getQfactor


function gettrangecorr(freqRange)
    dFreq = freqRange[2]-freqRange[1]
    freqMax = abs(freqRange[1])
    dt = 1 ./(2*freqMax)
    tMax = 0.5*dt.*size(freqRange,1)
    #return  -0.5 ./dFreq:1 ./(2*freqMax):0.5 ./dFreq
    return -tMax:dt:(tMax-dt)
end

function getcorrtime(freqRange, S; method="centerOfMassAbs")
    tRangeCorr = gettrangecorr(freqRange)
    dt = tRangeCorr[2] - tRangeCorr[1]
    Cx = ifftshift(ifft(ifftshift(S)))
    indt0 = findmin(abs.(tRangeCorr))[2]
    if method=="abs"
        Cx = Cx./Cx[indt0+1]
        dt.*sum(abs.(Cx[indt0+1:end]))
    elseif method=="square"
        Cx = Cx./Cx[indt0+1]
        dt.*sum((Cx[indt0+1:end]).^2)
    elseif method=="centerOfMassAbs"
        dt.*sum(tRangeCorr[indt0:end-1] .* abs.(Cx[indt0+1:end]))./(dt.*sum(abs.(Cx[indt0+1:end])))

    elseif method=="centerOfMassSquare"
        dt.*sum(tRangeCorr[indt0:end-1] .* (Cx[indt0+1:end]).^2 )./(dt.*sum((Cx[indt0+1:end]).^2))
    end
end
export getcorrtime

function getautocorrfromspectrum(freqRange, S)
    dFreq = freqRange[2]-freqRange[1]
    tRange = gettrangecorr(freqRange)
    return size(S,1).*dFreq .* ifftshift(ifft(ifftshift( real.(S) )) );
end
export getautocorrfromspectrum
