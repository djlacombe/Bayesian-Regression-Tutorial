using LinearAlgebra
using Distributions

function normal_linear_gibbs()
n = 1000
k = 2
sige = .5
x = randn(n,k)
beta = ones(k)*3
epsilon = randn(n)*sige
y = x*beta + epsilon
iter = 110000
burn = 10000
betas_final = zeros(iter-burn,k)
sigma_final = zeros(iter-burn,1)

for gibbs = 1:iter

# Draw for sigma
temp1 = y - x*beta
epe = dot(temp1,temp1)
temp2 = rand(InverseGamma(n,epe))
sigma = sqrt(temp2)

# Draw for Beta
betacov = inv((x'*x)*sigma^-2)
betamean = betacov*(x'*y*sigma^-2)
beta = rand(MvNormal(betamean,betacov))

# Save draws after burn in
   if gibbs>burn;
       betas_final[gibbs-burn,:] = beta';
       sigma_final[gibbs-burn,1] = sigma;
   end
end

betas = betas_final,sigma_final

mean_beta_1 = mean(betas_final[:,1])

mean_beta_2 = mean(betas_final[:,2])

mean_sigma = mean(sigma_final)

data = [mean_beta_1 mean_beta_2 mean_sigma]

print(data)

return betas

end
