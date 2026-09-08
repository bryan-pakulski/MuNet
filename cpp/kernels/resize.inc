int h=D[2],w=D[3],oh=A[0],ow=A[1];
if(MODE==0){int plane=i/(oh*ow),y=(i/ow%oh)*h/oh,x=(i%ow)*w/ow;return X(0,plane*h*w+y*w+x);}
int plane=i/(h*w),y=i/w%h,x=i%w;float result=0.0f;
int y0=(y*oh+h-1)/h,y1=((y+1)*oh+h-1)/h,x0=(x*ow+w-1)/w,x1=((x+1)*ow+w-1)/w;
for(int yy=y0;yy<y1;++yy)for(int xx=x0;xx<x1;++xx)result+=X(1,(plane*oh+yy)*ow+xx);return result;
