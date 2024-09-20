# 各组分相对分子质量
# μ=[2.01588 16.042 28.05 30.068 42.08 44.094 58.12 58.12 72.146 72.0146 86.172 100.198 114.224 178.25 142.276 156.3 28.016 44.01 34.076];

# 每个待求阀室处（pipe1共7个阀室）可以由连续性方程（正向+反向）得到两个表达式F，随后利用ipopt得到最合适的M：∑（|F|^2）最小
# for esti in 1:num_valve

# #    F1=( valve_density[2*esti+1,1]-initial_density[2*esti+1,1]+startpoint[1,4]-initial_density[1,1] ) + ratio/area*( M_guess[esti,2]+initial_M[2*esti+1,1]-initial_M[1,1]-startpoint[1,3] );
# #    F2=( endpoint[1,4]-initial_density[sn_total,1]+valve_density[esti+1,1]-initial_density[2*esti+1,1] ) + ratio/area*( endpoint[1,3]+initial_M[sn_total,1]-M_guess[esti,2]-initial_M[2*esti+1,1] );


#     goal = Model(Ipopt.Optimizer)
#     @variable(goal, M_guess[esti,i in 1:2]>=0)
#     @objective(goal, Min, (abs(( valve_density[2*esti+1,1]-initial_density[2*esti+1,1]+startpoint[1,4]-initial_density[1,1] ) + ratio/area*( M_guess[esti,2]+initial_M[2*esti+1,1]-initial_M[1,1]-startpoint[1,3] )))^2+(abs(( endpoint[1,4]-initial_density[sn_total,1]+valve_density[esti,1]-initial_density[2*esti+1,1] ) + ratio/area*( endpoint[1,3]+initial_M[sn_total,1]-M_guess[esti,2]-initial_M[2*esti+1,1] )))^2)
#     @constraint(goal, M_guess[esti,1]=M_guess[esti,2])
#     optimize!(goal)
#     return M_guess[esti,1]

#     M_profile[4*esti+1,1]=M_guess(esti,1);

# end








# 未知量，待求解，但不仅限于阀室处，包含所有自主划分的节点处的质量流量M
M_profile=zeros(sn_total,1);

