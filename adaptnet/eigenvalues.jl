import Base.atan

"
New method for the atan function to keep into account the quadrant that we are considering, similarly to what is done in Mathematica.
Computes the arctan of y./x
"
function atan(x,y)
    if x==0 && y==0
        return 0.
    else
        return atan(y./x) + (x<0)*pi
    end
end   
export atan

"
Build the matrix corresponding to the linearized adaptive network given
the weight matrix w.
"
function buildA(w; gamma=0.1, b=1.)
  N = size(w,1)
  vcat( hcat( -eye(N) + w , -eye(N) ), hcat( b.*gamma.*eye(N) , -gamma.*eye(N) ) )
end
export buildA

"
Given an eigenvalue of W, it returns the two corresponding eigenvalues of A, the linearized matrix of the corresponding adaptive network.
"
function transformLambda(gamma, beta, lambdaW)
    0.5*(-1+lambdaW-gamma +sqrt(complex((-1+lambdaW+gamma)^2 -4*gamma*beta))) , 
    0.5*(-1+lambdaW-gamma -sqrt(complex((-1+lambdaW+gamma)^2 -4*gamma*beta)))
end
export transformLambda

function realPartLambdaPlus(gamma, beta, g, phi)
    1 ./2 .*(-1 + 
   g*cos(phi) - gamma + ((g^2*cos(2*phi) + 
          2*(gamma - 1)*g*cos(phi) + (gamma - 1)^2 - 
          4*gamma*beta)^2 + (g^2*sin(2*phi) + 
          2*(gamma - 1)*g*sin(phi))^2)^(1 ./4)*
    cos(1 ./2 * 
      atan(g^2*cos(2*phi) + 
          2*(gamma - 1)*g*cos(phi) + (gamma - 1)^2 - 
          4*gamma*beta, 
          g^2*sin(2*phi) + 
          2*(gamma - 1)*g*sin(phi))))
end
function realPartLambdaMinus(gamma, beta, g, phi)
    1 ./2 .*(-1 + 
   g*cos(phi) - gamma - ((g^2*cos(2*phi) + 
          2*(gamma - 1)*g*cos(phi) + (gamma - 1)^2 - 
          4*gamma*beta)^2 + (g^2*sin(2*phi) + 
          2*(gamma - 1)*g*sin(phi))^2)^(1 ./4)*
    cos(1 ./2 * 
      atan(g^2*cos(2*phi) + 
          2*(gamma - 1)*g*cos(phi) + (gamma - 1)^2 - 
          4*gamma*beta,
            g^2*sin(2*phi) + 
          2*(gamma - 1)*g*sin(phi))))
end


"
Performs a binary search for the critical g given gamma and beta.
In case of convergence, it returns the critical g, the phase of the corresponding non-adaptive eigenvalue and in which branch (plus or minus), it is found. 
"
function findcriticalg(gamma,beta; g0=1.0, epsilon=1e-3, maxIter=100, gMin=0., gMax=5., verbose=false)

    gTmp = copy(g0)
    maxRe = (-1,0)
    maxPlus = (-1,0)
    maxMinus = (-1,0)
    iter = 0
    converged = false

    while abs(maxRe[1])>epsilon && iter<maxIter
        # I think we can look at only half circle, because eigenvalues are complex conjugates
        maxPlus = findmax(realPartLambdaPlus.(gamma, beta, gTmp, 0.:0.01:pi))
        maxMinus = findmax(realPartLambdaMinus.(gamma, beta, gTmp, 0.:0.01:pi))
        maxRe = findmax([maxPlus[1], maxMinus[1]])
        if verbose
            println("iter: ",iter," | g: ",gTmp," maxRe: ",maxRe)
        end
        if maxRe[1]>epsilon
            gMax = copy(gTmp)
            gTmp = 0.5*(gMin+gTmp) 
        elseif maxRe[1]<-epsilon
            gMin = copy(gTmp)
            gTmp = 0.5*(gMax+gTmp)
        end
        iter+=1
    end
    if abs(maxRe[1])<=epsilon
        if maxRe[2]==1
            phiMax = (0.:0.01:pi)[maxPlus[2]]
            maxSide = maxRe[2]
        elseif maxRe[2]==2
            phiMax = (0.:0.01:pi)[maxMinus[2]]
            maxSide = maxRe[2]
        end
        return gTmp, phiMax, maxSide, true
    else
        return false
    end
end
export findcriticalg

"
Returns the critical g derived from the condition that the linear susceptibility
is equal to 1 at its max, in the Fourier domain.
"
function getcriticalg(gamma, beta)
  # check reality of the second stationary point #
  if gamma^2 <= sqrt(beta*gamma^2*(beta+2+2*gamma))
    return sqrt(1-gamma*(2*beta+gamma) + 2*sqrt(beta*gamma^2*(2+beta+2*gamma)))
  else
    return 1 + beta
  end
end
export getcriticalg

function getbetahopf(gamma)
  -1 -gamma +sqrt(1+2*gamma+2*gamma^2)
end

function getgammahopf(b)
  b + sqrt(2*(b^2+b))
end
export getbetahopf, getgammahopf