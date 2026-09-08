int ax=A[0],inner=SO(ax),q=i/inner%O[ax],outer=i/(inner*O[ax]),start=0;
for(int a=0;a<IN;++a){int width=D[a*8+ax];if(q<start+width)return X(a,(outer*width+q-start)*inner+i%inner);start+=width;}
return 0.0f;
