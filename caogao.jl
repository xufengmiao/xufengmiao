include("新压缩机1.jl")
include("BWRS.jl")
include("pipe.jl")
include("xiaxian.jl")

x  = [0.975 0 0.002 0 0.002 0 0 0 0 0 0 0 0 0 0 0.016 0.005 0];#气体组分
Qs = 4550*(10^4)/(60*60*24)/2#体积流量m³/s
#管段：
Ke = 0.0177e-3;#粗糙度m
D = 1.016;#外径m
d = 0.9812;#内径m
Th = 278;#环境温度K
GDZC = [160e3 0 160e3 0 160e3 0 160e3 100e3 80e3 80e3];#管段长度m
L = 2000;#步长m
crxs = 1.15;#W/(㎡·K)
P_last=5e6#管道终点压力
P_xiaxian=6.5e6#管道的进口压力下限

#压气站:
nmin = [6000 0 5160 0 5160 0 0 5160 0 0 5160];#转速可行区间r/min
nmax = [10500 0 9030 0 9030 0 0 9030 0 0 9030];
ε=0.001;#精度
K=[4 0 2 0 2 0 0 2 0 0 2];#每个压气站的压缩机台数
n0=[8000 0 9030 0 9030 0 0 9030 0 0 9030];#额定转速

a1=[-1423.7 0 -243.41 0 -243.41 0 0 -243.41 0 0 -243.41];
b1=[4713 0 684.18 0 684.18 0 0 684.18 0 0 684.18];
c1=[6635.9 0 8677.2 0 8677.2 0 0 8677.2 0 0 8677.2];
a2=[-13.91 0 -2.38 0 -2.38 0 0 -2.38 0 0 -2.38];
b2=[68.03 0 18.96 0 18.96 0 0 18.96 0 0 18.96];
c2=[2.08 0 47.72 0 47.72 0 0 47.72 0 0 47.72];#多项式系数

D1=126.51*10^4/(60*60*24);
D2=126.52*10^4/(60*60*24);
D3=253.02*10^4/(60*60*24);
UGS=D1+D2+D3;

(P1_min,P2_min,P3_min,P4_min,P5_min)=xiaxian(P_last,P_xiaxian)
a=P1_min/1e6
b=P2_min/1e6
c=P3_min/1e6
f=P4_min/1e6
e=P5_min/1e6


#第一个压气站
#while Pc>6.5
x1=rand(a*10000:10*10000)/10000
Pr=5e6;#压缩机进口压力
T=293;
Pdr=x1*1e6
(mingl,T_qiu1,kj,gl)=ysj(Pr,Pdr,T,Qs,n0[1],nmin[1],nmax[1],x,a1[1],b1[1],c1[1],a2[1],b2[1],c2[1],K[1])
f1=mingl[1]
K1=kj[1]
#一二压气站之间的管道系统
if f1==0
  Pj=Pr
  Pj=T
  K1=0
else
  Pj=x1*1e6;
  Tj=T_qiu1[1];
end
if Tj>333
  Tj=333
end
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[1],L,crxs,x)
if Pc<6.5e6
  f1=1000
end
#end


#第二个压气站
#while Pc>6.5
x2=rand(b*10000:10*10000)/10000
Pr=Pc
T=Tc
Pdr=x2*1e6;
(mingl,T_qiu1,kj,gl)=ysj(Pr,Pdr,T,Qs,n0[3],nmin[3],nmax[3],x,a1[3],b1[3],c1[3],a2[3],b2[3],c2[3],K[3])
f2=mingl[1]
K2=kj[1]
#二三压气站之间的管道系统
if f2==0
  Pj=Pr
  Tj=T
  K2=0
else
  Pj=x2*1e6;
  Tj=T_qiu1[1];
end
if Tj>333
  Tj=333
end
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[3],L,crxs,x)
if Pc<6.5e6
  f2=1000
end
#end


#第三个压气站
#while Pc>6.5
x3=rand(c*10000:10*10000)/10000
Pr=Pc;
T=Tc;
Pdr=x3*1e6;
(mingl,T_qiu1,kj,gl)=ysj(Pr,Pdr,T,Qs,n0[5],nmin[5],nmax[5],x,a1[5],b1[5],c1[5],a2[5],b2[5],c2[5],K[5])
f3=mingl[1]
K3=kj[1]


#三四压气站之间的管道系统
if f3==0
  Pj=Pr
  Tj=T
  K3=0
else
  Pj=x3*1e6;
  Tj=T_qiu1[1];
end
if Tj>333
  Tj=333
end
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[5],L,crxs,x)
if Pc<6.5e6
  f3=1000
end


#第四个压气站
#while Pc>6.5
x4=rand(e*10000:10*10000)/10000
Pr=Pc;
T=Tc;
Pdr=x4*1e6;
Qs=Qs-D1;
(mingl,T_qiu1,kj,gl)=ysj(Pr,Pdr,T,Qs,n0[8],nmin[8],nmax[8],x,a1[8],b1[8],c1[8],a2[8],b2[8],c2[8],K[8])
f4=mingl[1]
K4=kj[1]


#四五压气站之间的管道系统
if f4==0
  Pj=Pr
  Tj=T
  K4=0
else
  Pj=x4*1e6;
  Tj=T_qiu1[1];
end
if Tj>333
  Tj=333
end
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[7],L,crxs,x)
if Pc<6.5e6
  f4=1000
end


#第五个压气站
#while P_last>5
x5=rand(f*10000:10*10000)/10000
Pr=Pc;
T=Tc;
Pdr=x5*1e6;
Qs=Qs-D2;
(mingl,T_qiu1,kj,gl)=ysj(Pr,Pdr,T,Qs,n0[11],nmin[11],nmax[11],x,a1[11],b1[11],c1[11],a2[11],b2[11],c2[11],K[11])
f5=mingl[1]
K5=kj[1]
if f5==0
  Pj=Pr
  Tj=T
  K5=0
else
  Pj=x5*1e6;
  Tj=T_qiu1[1];
end
if Tj>333
  Tj=333
end
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[8],L,crxs,x)
Pj=Pc;
Tj=Tc;
if Tj>333
  Tj=333
end
Qs=Qs-D3;
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[9],L,crxs,x)
Pj=Pc;
Tj=Tc;
if Tj>333
  Tj=333
end
Qs=Qs+UGS;
(Pc,Tc)=pipe(Pj,Tj,Qs,Ke,D,d,Th,GDZC[10],L,crxs,x)
P_last=Pc
#end


F=f1+f2+f3+f4+f5


# return F,x1,x2,x3,x4,x5,K1,K2,K3,K4,K5
# end
