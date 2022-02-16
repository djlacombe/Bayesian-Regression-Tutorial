# Gibbs sampler for hte normal linear model
# Call the function using the following:
# include("normal_linear_gibbs.jl")
# betas,sigma = normal_linear_gibbs()

# Required packages

using LinearAlgebra
using Distributions

# Main function
function normal_linear_gibbs()

n = 1000    # Number of observations
k = 2       # Number of explanatory variables
sige = .5   # Error variance
x = randn(n,k)     # Explanatory variables
beta = ones(k)*3   # Value of coefficients
epsilon = randn(n)*sige # Error term
y = x*beta + epsilon   # Generate dependent variable 
gibbs_samples = 11000   # Number of Gibbs samples
burn_in = 1000     # Number of burn in draws
betas_final = zeros(gibbs_samples-burn_in,k)    # Storage for Gibbs samples of betas
sigma_final = zeros(gibbs_samples-burn_in,1)    # Storage for Gibbs samples of sigma

# Main loop

for i = 1:gibbs_samples

# Draw for sigma
temp1 = y - x*beta
epe = dot(temp1,temp1)
temp2 = rand(InverseGamma(n,epe))
sigma = sqrt(temp2)

# Draw for Beta
betacov = inv(x'*x)*sigma^2
betamean = inv(x'*x)*x'*y
beta = rand(MvNormal(betamean,betacov))

# Save draws after burn in
   if i>burn_in;
       betas_final[i-burn_in,:] = beta';
       sigma_final[i-burn_in,1] = sigma;
   end
end

return betas_final, sigma_final

end
