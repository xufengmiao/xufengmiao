using JuMP
using Ipopt
gp = Model(Ipopt.Optimizer)
@variable(gp, P[i in 1:2]>=0)
@objective(gp, Min, 3*P[1]+2*P[2])
@constraint(gp, P[1]-P[2]>=0)
@constraint(gp, P[1]-5*P[2]+5>=0)
@constraint(gp, 2*P[1]+3*P[2]<=12)
optimize!(gp)
