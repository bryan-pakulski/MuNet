int ax=A[0],inner=SD(0,ax),width=D[ax],kdim=A[1];
if(MODE==0){int col=i/inner%width,base=i-col*inner;float value=X(0,i);int rank=0;for(int k=0;k<width;++k){float v=X(0,base+k*inner);if(v>value||(v==value&&k<col))++rank;}return float(rank);}
int rank=i/inner%kdim,base=i/(inner*kdim)*inner*width+i%inner;
for(int k=0;k<width;++k)if(X(1,base+k*inner)==float(rank))return float(k);return -1.0f;
