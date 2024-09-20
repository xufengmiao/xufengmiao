    using JuMP
    using Ipopt
    # w1 = 0.01
    # Number1 = 2
    # Number2 = 1
    # Number3 = 0
function multioptimization(Number1,Number2,Number3,w1)
    # w1 = 1
    # Number1 = 0
    # Number2 = 0
    # Number3 = 0
    d=1.016-17.4*2*10^-3;
    D=1.016;
    reden=0.5654;
    Ke=0.017*10^-3;
    T_0=278;
    A01=200.326827108680;B01=0.0444;C01=2212409.40302635;D01=37369729.1764816;E01=1308725230.88976;
    aa1=7.7925;bb1=0.0053;cc1=302203.851412505;dd1=241.3268;rr1=0.0054;arr1=6.90617229223050*10^-5;
    ZMui=16.3776;
    Rm = 8.3143;
    nn=56;
    space=10000;
    gp = Model(Ipopt.Optimizer)
    #----------------------defining variable-------------------------
    Pl = 5*10^3*ones(nn)
    if Number1> 0
        Pl[18]=6.5*10^3
    end
    if Number1> 0 || Number2> 0
        Pl[35]=6.5*10^3
    end
    @variable(gp, Pl[i]<=P[i in 1:nn]<=10^4) #defining pressure variables 2nd stage
    @variable(gp, Q>=0) #defining delivery rate
    @variable(gp, power>=0) #defining delivery rate
    @variable(gp, den[1:nn]>=0,start=4) #defining density variables 密度
    @variable(gp, 0<=Z[1:nn]<=1,start=1) #defining compressbility variables
    @variable(gp, lam[1:nn]>=0,start=10) #defining friction variables 摩擦
    @variable(gp, 0<=Di[1:nn]<=10,start=3) #defining variables
    @variable(gp, 1<=kv[1:3]<=10,start=1.4) #defining variables
    @variable(gp, 0.8<=T[1:nn]<=1.1978,start=1) #defining temperatures 温度范围（T/T0）
    @variable(gp, 0<=arr[1:nn]<=100,start=3) #defining variables 变量
    @variable(gp, 20<=Cpp[1:nn]<=80,start=40) #defining variables 压力绝热指数
    @variable(gp, 0<=Cv[1:3]<=50,start=27) #defining variables 体积绝热指数
    @variable(gp, hed[1:3]>=0,start=9000) #defining head of each compressor station  压头
    @variable(gp, 6000<=rspeed1<=10500, start=10500) #defining rotational speed 1 转速
    @variable(gp, 5521<=rspeed2<=9662, start=9600) #defining rotational speed 2
    @variable(gp, 5521<=rspeed3<=9662, start=9600) #defining rotational speed 3
    @variable(gp, 0<=pout[1:3]<=10^4, start=9600) #defining discharge pressure 出口压力
    @variable(gp, 0<=xiaolv[1:3]<=100, start=85) #效率
    #----------------------Objective-------------------------目标
    # @NLobjective(gp, Min, (4*100*hed[1]*9.8*(Q*101325*ZMui/1000/293.15/Rm)/1000/xiaolv[1] +
    # 4*100*hed[1]*9.8*(Q*101325*ZMui/1000/293.15/Rm)/1000/xiaolv[1] +
    # 4*100*hed[1]*9.8*(Q*101325*ZMui/1000/293.15/Rm)/1000/xiaolv[1])*12/10^4/200)
    #w1 = 0.01
    @NLobjective(gp, Max, Q)

    @NLconstraint(gp, (Number1*100*hed[1]*9.8*(Q*101325*ZMui/1000/293.15/Rm)/1000/xiaolv[1] +
    Number2*100*hed[2]*9.8*(Q*101325*ZMui/1000/293.15/Rm)/1000/xiaolv[2] +
    Number3*100*hed[3]*9.8*(Q*101325*ZMui/1000/293.15/Rm)/1000/xiaolv[3])*12/10^4/200 - power == 0)
#----------------------pyhsical parameter of natural gas-------------------------天然气物理参数
    @NLexpression(gp, p_T[i=1:nn], den[i]*Rm+(B01*Rm+2*C01/(T[i]*T_0)^3-3*D01/(T[i]*T_0)^4+4*E01/(T[i]*T_0)^5)*den[i]^2+
    (bb1*Rm+dd1/(T[i]*T_0)^2)*den[i]^3-(arr1*dd1/(T[i]*T_0)^2)*den[i]^6-(2*cc1*den[i]^3/(T[i]*T_0)^3)*(1+rr1*den[i]^2)*exp(-1*rr1*den[i]^2));

    @NLexpression(gp, p_Rou[i=1:nn], Rm*(T[i]*T_0)+2*(B01*Rm*(T[i]*T_0)-A01-C01/(T[i]*T_0)^2+D01/(T[i]*T_0)^3-E01/(T[i]*T_0)^4)*den[i]+
    3*(bb1*Rm*(T[i]*T_0)-aa1-dd1/(T[i]*T_0))*den[i]^2+6*arr1*(aa1+dd1/(T[i]*T_0))*den[i]^5+(3*cc1*den[i]^2/(T[i]*T_0)^2)*(1+rr1*den[i]^2-2*rr1^2*den[i]^4/3)*exp(-1*rr1*den[i]^2));

    @NLconstraint(gp, [i = 1:nn], den[i]*Rm*T[i]*T_0+(B01*Rm*(T[i]*T_0)-A01-C01/((T[i]*T_0)^2)+D01/((T[i]*T_0)^3)-E01/((T[i]*T_0)^4))*(den[i]^2)+(bb1*Rm*(T[i]*T_0)-aa1-dd1/(T[i]*T_0))*(den[i]^3)+
    arr1*(aa1+dd1/(T[i]*T_0))*(den[i]^6) +(cc1*(den[i]^3)/((T[i]*T_0)^2))*(1+rr1*(den[i]^2))*exp(-rr1*(den[i]^2))-P[i] == 0)

    @NLconstraint(gp, [i = 1:nn], den[i]*Rm*(T[i]*T_0)*Z[i]==P[i])
    #----------------------pipe segment pressure vs flow rate-------------------------管段压力与流量
    # pipe segment #1
    @NLconstraint(gp, [i=3:18], lam[i-1]+2*log10(Ke/3.7/d+2.51*lam[i-1]/(1.536*Q*reden/d/10^-5))==0) #friction coefficient
    @NLconstraint(gp, [i=3:18], 0.001*(P[i]^2-(P[i-1]^2-(Q)^2*Z[i-1]*reden*space*T[i-1]*T_0/lam[i-1]^2/0.03848^2/d^5/10^6))==0);

    #pipe segment #2
    @NLconstraint(gp, [i=20:35], lam[i-1]+2*log10(Ke/3.7/d+2.51*lam[i-1]/(1.536*Q*reden/d/10^-5))==0) #friction coefficient
    @NLconstraint(gp, [i=20:35], 0.001*(P[i]^2-(P[i-1]^2-(Q)^2*Z[i-1]*reden*space*T[i-1]*T_0/lam[i-1]^2/0.03848^2/d^5/10^6))==0);

    # #pipe segment #3
    @NLconstraint(gp, [i=37:56], lam[i-1]+2*log10(Ke/3.7/d+2.51*lam[i-1]/(1.536*Q*reden/d/10^-5))==0) #friction coefficient
    @NLconstraint(gp, [i=37:56], 0.001*(P[i]^2-(P[i-1]^2-(Q)^2*Z[i-1]*reden*space*T[i-1]*T_0/lam[i-1]^2/0.03848^2/d^5/10^6))==0);

    #----------------------pipe segment temperature drop-------------------------管段温降

    @NLconstraint(gp, [i=1:nn], 13.19+0.092*(T[i]*T_0)-(6.24*10^-5)*(T[i]*T_0)^2+5.1129*10^-4*(P[i]^1.124)/(T[i]^5.08)-Cpp[i]==0);
    @NLconstraint(gp,[i=2:nn+1],0.98*10^6-1.5*((T[i-1]*T_0)^2)-Di[i-1]*(Cpp[i-1]/ZMui)*((T[i-1]*T_0)^2)==0);

    #pipe segment #1
    @NLconstraint(gp,[i=2:18],1.1*3.14159*D*10^6/(101325*ZMui/1000/293.15/Rm)/(1000/ZMui)/arr[i]-Cpp[i]*(Q)==0);
    @NLconstraint(gp,[i=3:18],1+(T[i-1]-1)*exp(-arr[i-1]*space/10^6)-T[i]-Di[i-1]*0.001*(P[i-1]-P[i])*(1-exp(-arr[i-1]*space/10^6))/(arr[i-1]*space/10^6)/T_0==0);

    #pipe segment #2
    @NLconstraint(gp,[i=19:35],1.1*3.14159*D*10^6/(101325*ZMui/1000/293.15/Rm)/(1000/ZMui)/arr[i]-Cpp[i]*(Q)==0);
    @NLconstraint(gp,[i=20:35],1+(T[i-1]-1)*exp(-arr[i-1]*space/10^6)-T[i]-Di[i-1]*0.001*(P[i-1]-P[i])*(1-exp(-arr[i-1]*space/10^6))/(arr[i-1]*space/10^6)/T_0==0);

    # #pipe segment #3
    @NLconstraint(gp,[i=36:56],1.1*3.14159*D*10^6/(101325*ZMui/1000/293.15/Rm)/(1000/ZMui)/arr[i]-Cpp[i]*(Q)==0);
    @NLconstraint(gp,[i=37:56],1+(T[i-1]-1)*exp(-arr[i-1]*space/10^6)-T[i]-Di[i-1]*0.001*(P[i-1]-P[i])*(1-exp(-arr[i-1]*space/10^6))/(arr[i-1]*space/10^6)/T_0==0);
    #----------------------compressor station-------------------------压缩机站
    # compressor station 1,Pin=5MPa,Tin=293K,Z=0.9082
    if Number1 > 0
        @NLconstraint(gp, -13.91*((Q*101325*Z[1]*T[1]*T_0/(P[1]*1000)/293.15))^2*(8000/rspeed1)^2+ 68.03*Number1*((Q*101325*Z[1]*T[1]*T_0/(P[1]*1000)/293.15))*(8000/rspeed1) + 2.08*Number1^2 - Number1^2*xiaolv[1]==0)

        @NLconstraint(gp,((Q*101325*Z[1]*T[1]*T_0/(P[1]*1000)/293.15))*8000-Number1*1.79615*rspeed1>=0)

        @NLconstraint(gp,Number1*3.6336*rspeed1-((Q*101325*Z[1]*T[1]*T_0/(P[1]*1000)/293.15))*8000>=0)

        @NLconstraint(gp,-1423.7*((Q*101325*Z[1]*T[1]*T_0/(P[1]*1000)/293.15))^2+ 4713*Number1*((Q*101325*Z[1]*T[1]*T_0/(P[1]*1000)/293.15))*rspeed1/8000 + Number1^2*6635.9*(rspeed1/8000)^2-Number1^2*hed[1]==0)

        @constraint(gp,T[2]-323/T_0==0)

        @NLconstraint(gp,pout[1]-P[1]*((1+(kv[1]-1)*9.8*ZMui*hed[1]/kv[1]/Z[1]/Rm/(T[1]*T_0)/1000)^(kv[1]/(kv[1]-1)))==0)

        @NLconstraint(gp, den[1]*Cpp[1]*p_Rou[1]-kv[1]*Cv[1]*P[1]==0)

        @NLconstraint(gp, Cv[1]+(T[1]*T_0/den[1]^2)*(p_T[1]^2)/p_Rou[1]-Cpp[1]==0);

        @constraint(gp,P[2]-pout[1]==0)

    elseif Number1 == 0
        @NLconstraint(gp,hed[1]==0)

        @constraint(gp,T[2]-T[1]==0)

        @NLconstraint(gp,P[2]-P[1]==0)
    end

    # compressor station 2,Pin=P81,Tin=T81
    if Number2 > 0
        @NLconstraint(gp, -3.17*((Q*101325*Z[18]*T[18]*T_0/(P[18]*1000)/293.15))^2*(8626.75/rspeed2)^2+ 23.83*Number2*((Q*101325*Z[18]*T[18]*T_0/(P[18]*1000)/293.15))*(8626.75/rspeed2) + Number2^2*42.14- Number2^2*xiaolv[2]==0)

        @NLconstraint(gp,((Q*101325*Z[18]*T[18]*T_0/(P[18]*1000)/293.15))*8626.75-Number2*2.961*rspeed2>=0)

        @NLconstraint(gp,Number2*5.4498*rspeed2-(Q*101325*Z[18]*T[18]*T_0/(P[18]*1000)/293.15)*8626.75>=0)

        @NLconstraint(gp,-275.89*((Q*101325*Z[18]*T[18]*T_0/(P[18]*1000)/293.15))^2+ 1400.3*Number2*((Q*101325*Z[18]*T[18]*T_0/(P[18]*1000)/293.15))*rspeed2/8626.75 + Number2^2*4602.9*(rspeed2/8626.75)^2-Number2^2*hed[2]==0)

        @NLconstraint(gp,pout[2]-P[18]*((1+(kv[2]-1)*9.8*ZMui*hed[2]/kv[2]/Z[18]/Rm/(T[18]*T_0)/1000)^(kv[2]/(kv[2]-1)))==0)

        @NLconstraint(gp, den[18]*Cpp[18]*p_Rou[18]-kv[2]*Cv[2]*P[18]==0)

        @NLconstraint(gp, Cv[2]+(T[18]*T_0/den[18]^2)*(p_T[18]^2)/p_Rou[18]-Cpp[18]==0);

        @constraint(gp,P[19]-pout[2]==0)

        @constraint(gp,T[19]-323/T_0==0)

    elseif Number2 == 0
        @NLconstraint(gp,hed[2]==0)

        @NLconstraint(gp,P[18]-P[19]==0)

        @constraint(gp,T[18]-T[19]==0)
    end

    # compressor station 3,Pin=P172,Tin=T172
    if Number3 > 0
        @NLconstraint(gp, -3.17*((Q*101325*Z[35]*T[35]*T_0/(P[35]*1000)/293.15))^2*(8626.75/rspeed3)^2+ 23.83*Number3*((Q*101325*Z[35]*T[35]*T_0/(P[35]*1000)/293.15))*(8626.75/rspeed3) + Number3^2*42.14- Number3^2*xiaolv[3]==0)

        @NLconstraint(gp,((Q*101325*Z[35]*T[35]*T_0/(P[35]*1000)/293.15))*8626.75-Number3*2.961*rspeed3>=0)

        @NLconstraint(gp,Number3*5.4498*rspeed3-((Q*101325*Z[35]*T[35]*T_0/(P[35]*1000)/293.15))*8626.75>=0)

        @NLconstraint(gp,-275.89*((Q*101325*Z[35]*T[35]*T_0/(P[35]*1000)/293.15))^2+ 1400.3*Number3*((Q*101325*Z[35]*T[35]*T_0/(P[35]*1000)/293.15))*rspeed3/8626.75 + Number3^2*4602.9*(rspeed3/8626.75)^2-Number3^2*hed[3]==0)

        @NLconstraint(gp,pout[3]-P[35]*((1+(kv[3]-1)*9.8*ZMui*hed[3]/kv[3]/Z[35]/Rm/(T[35]*T_0)/1000)^(kv[3]/(kv[3]-1)))==0)

        @NLconstraint(gp, den[35]*Cpp[35]*p_Rou[35]-kv[3]*Cv[3]*P[35]==0)

        @NLconstraint(gp, Cv[3]+(T[35]*T_0/den[35]^2)*(p_T[35]^2)/p_Rou[35]-Cpp[35]==0);

        @constraint(gp,P[36]-pout[3]==0)

        @constraint(gp,T[36]-333/T_0==0)

    elseif  Number3 == 0
        @NLconstraint(gp,hed[3]==0)

        @constraint(gp,P[35]-P[36]==0)

        @constraint(gp,T[35]-T[36]==0)
    end

    # source boundary condition
    @constraint(gp,P[1]-5500==0)

    @constraint(gp,T[1]-293/T_0==0)

    optimize!(gp)

    if termination_status(gp) == LOCALLY_SOLVED
        return w1*value(Q)-(1-w1)*value(power), value(Q), value(power)
    else
        return -0, 0, 0
    end
end
