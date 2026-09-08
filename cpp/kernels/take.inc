int ax=A[0],inner=SD(0,ax),width=D[ax],indices=N[1];
if(MODE==0){int k=i/inner%indices;float v=X(1,k);if(!isfinite(v)||v<0.0f||v>=float(width)||floor(v)!=v)return 0.0f;return X(0,(i/(inner*indices)*width+int(v))*inner+i%inner);}
int col=i/inner%width,base=i/(inner*width)*inner*indices+i%inner;float result=0.0f;
for(int k=0;k<indices;++k)if(X(1,k)==float(col))result+=X(2,base+k*inner);return result;
