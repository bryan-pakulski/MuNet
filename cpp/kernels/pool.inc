int h=D[2],w=D[3],kh=A[0],kw=A[1],sy=A[2],sx=A[3],py=A[4],px=A[5];
int oh=MODE<2?O[2]:D[10],ow=MODE<2?O[3]:D[11];
bool average=MODE==1||MODE==3;
int plane=MODE<2?i/(oh*ow):i/(h*w),iy=MODE<2?0:i/w%h,ix=MODE<2?0:i%w;
int ylo=MODE<2?i/ow%oh:max(0,(iy+py-kh)/sy),yhi=MODE<2?ylo+1:min(oh,(iy+py)/sy+1);
int xlo=MODE<2?i%ow:max(0,(ix+px-kw)/sx),xhi=MODE<2?xlo+1:min(ow,(ix+px)/sx+1);
float result=0.0f;
for(int oy=ylo;oy<yhi;++oy)for(int ox=xlo;ox<xhi;++ox){
  int y0=oy*sy-py,x0=ox*sx-px,y1=min(y0+kh,h),x1=min(x0+kw,w);
  int count=A[7]!=0?(min(y0+kh,h+py)-max(y0,-py))*(min(x0+kw,w+px)-max(x0,-px)):(y1-max(y0,0))*(x1-max(x0,0));
  float value=average?0.0f:-3.402823466e38f;int winner=-1;
  for(int y=max(y0,0);y<y1;++y)for(int x=max(x0,0);x<x1;++x){float v=X(0,plane*h*w+y*w+x);if(average)value+=v;else if(winner<0||v>value){value=v;winner=y*w+x;}}
  if(MODE<2)return average?value/float(count):value;
  if(iy>=y0&&iy<y1&&ix>=x0&&ix<x1&&(average||winner==iy*w+ix))result+=X(1,(plane*oh+oy)*ow+ox)/(average?float(count):1.0f);
}
return result;
