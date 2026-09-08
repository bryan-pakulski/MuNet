int j=0;
for(int d=0;d<RO;++d){int q=i/SO(d)%O[d];if(MODE==0)j+=(A[d]+q*A[2*RO+d])*SD(0,d);else{int delta=q-A[d],step=A[2*RO+d];if(delta%step!=0)return 0.0f;int c=delta/step;if(c<0||c>=A[RO+d])return 0.0f;j+=c*SD(0,d);}}
return X(0,j);
