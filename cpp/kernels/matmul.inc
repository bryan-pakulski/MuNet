int m=O[RO-2],n=O[RO-1],kdim=D[R[0]-1],row=i/n%m,col=i%n;
int ab=0,bb=0;
for(int d=0;d<R[0]-2;++d)if(D[d]!=1)ab+=(i/SO(d+RO-R[0])%O[d+RO-R[0]])*SD(0,d);
for(int d=0;d<R[1]-2;++d)if(D[8+d]!=1)bb+=(i/SO(d+RO-R[1])%O[d+RO-R[1]])*SD(1,d);
float result=0.0f;for(int k=0;k<kdim;++k)result+=X(0,ab+row*kdim+k)*X(1,bb+k*n+col);return result;
