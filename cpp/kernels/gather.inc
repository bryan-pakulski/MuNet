int ax=A[0],inner=SD(0,ax),width=D[ax],kdim=D[8+ax];
if(MODE==0){float v=X(1,i);if(!isfinite(v)||v<0.0f||v>=float(width)||floor(v)!=v)return 0.0f;int base=i/(inner*kdim)*inner*width+i%inner;return X(0,base+int(v)*inner);}
int col=i/inner%width,base=i/(inner*width)*inner*kdim+i%inner;float result=0.0f;
for(int k=0;k<kdim;++k)if(X(1,base+k*inner)==float(col))result+=X(2,base+k*inner);return result;
