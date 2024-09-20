function ysj(Pr,Pdr,T,Qs,n0,nmin,nmax,x,a1,b1,c1,a2,b2,c2,K)
    gl=[]   #功率
    mingl=[]
    kj=[]   #开机台数
    T_qiu=[]
    T_qiu1=[]

    for i in 1:K #开机台数

        include("BWRS.jl")

        P=1e-3Pr        #换算后的进口压力
        Pd=1e-3Pdr      #换算后的出口压力

        (Z,Cpm,Mg,Di,midu1,Kt,Kv)=BWRS(P,T,x)

        #实际入口状态下流量,m^3/s
        Q=Qs*0.101325*1e3*Z*T/(P*293.15);
        R=8.3143;#KJ/(Kmol·K)
        Rt=R/Mg#J/(g·K)#Mg:g/mol

        #实际压头
        H=Kv*Z*Rt*T*((Pd/P)^((Kv-1)/Kv)-1)/(Kv-1)#KJ/Kg

        #所求转速
        n1=2*n0*(a1*(Q/i)^2-102*H)/(-b1*Q/i-sqrt((b1*Q/i)^2-4*c1*(a1*(Q/i)^2-102*H)))  #n0为确定拟合曲线系数时的转速   #这里的102是1000/9.8
                                                                                       #其中的a1、b1、c1为转速拟合后的系数


        if n1<nmin||n1>nmax #喘振or滞止，压缩机不工作

            push!(T_qiu,T)
            push!(gl,10000)

        else
            
            Q0=Q*n0/(n1*i)  # 由于转速差别，在实际工况下，使用相似定律转化后，单台压缩机的进口压力
            η0=a2*Q0^2+b2*Q0+c2;   #其中a2、b2、c2为效率拟合后的转速
            η=η0/100;
            #压缩机温度多变指数
            mT=Kt*η/(Kt*η-(Kt-1));
            #出口温度
            T1=T*((Pd/P)^((mT-1)/mT))
            push!(T_qiu,T1)
            #压缩机电机的功率
            G=Q0*midu1*Mg; #Kg/s #Kmol/m^3 #g/mol
            W=H*G/(1000*η*0.95); #Mw
            W=W*i

                    if W<0
                        W=10000
                    end

            push!(gl,W)

        end

        

        if i==K
            min=gl[1]
            djw=1
            m=0
            for j in 1:K
                if min>gl[j]
                    min=gl[j]
                    djw=j
                end
                m+=gl[j]
                if j==K
                    if m==10000*K
                    min=0
                    end
                end
            end
            push!(mingl,min)
            push!(kj,djw)
            push!(T_qiu1,T_qiu[djw])
        end


  
        
    end
    return mingl,T_qiu1,kj,gl
end
 #进口压力Pr，出口压力Pdr，单位kpa
    #n0额定转速，nmin最小转速，nmax最大转速
    #Qs入口标况流量，T入口温度，x为组分
    #a1，a2，c1为压头-流量的系数
    #b1，b2，c2为效率-流量的系数
    #K为压缩机台数