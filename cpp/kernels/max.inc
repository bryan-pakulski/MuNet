int count=1;for(int a=0;a<NA;++a)count*=D[A[a]];
float result=-3.402823466e38f;
for(int k=0;k<count;++k){int j=0,red=count;for(int d=0;d<R[0];++d){bool reduced=false;for(int a=0;a<NA;++a)if(A[a]==d)reduced=true;if(reduced){red/=D[d];j+=(k/red%D[d])*SD(0,d);}else j+=(i/SO(d)%O[d])*SD(0,d);}result=max(result,X(0,j));}
return result;
