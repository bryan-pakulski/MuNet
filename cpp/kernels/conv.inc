int batch=D[0],ci=D[1],h=D[2],w=D[3],co=D[8],cg=D[9],kh=D[10],kw=D[11];
int sy=A[0],sx=A[1],py=A[2],px=A[3],dy=A[4],dx=A[5],groups=A[6],og=co/groups;
int oh=MODE==0?O[2]:D[18],ow=MODE==0?O[3]:D[19];
float result=0.0f;
if(MODE==0){
  int ox=i%ow,oy=i/ow%oh,oc=i/(ow*oh)%co,b=i/(ow*oh*co),group=oc/og;
  for(int c=0;c<cg;++c)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx){int iy=oy*sy-py+ky*dy,ix=ox*sx-px+kx*dx;if(iy>=0&&iy<h&&ix>=0&&ix<w)result+=X(0,((b*ci+group*cg+c)*h+iy)*w+ix)*X(1,((oc*cg+c)*kh+ky)*kw+kx);}
}else if(MODE==1){
  int ix=i%w,iy=i/w%h,c=i/(w*h)%ci,b=i/(w*h*ci),group=c/cg;
  for(int oc=group*og;oc<(group+1)*og;++oc)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx){int yy=iy+py-ky*dy,xx=ix+px-kx*dx;if(yy<0||xx<0||yy%sy!=0||xx%sx!=0)continue;int oy=yy/sy,ox=xx/sx;if(oy<oh&&ox<ow)result+=X(2,((b*co+oc)*oh+oy)*ow+ox)*X(1,((oc*cg+c%cg)*kh+ky)*kw+kx);}
}else{
  int kx=i%kw,ky=i/kw%kh,c=i/(kw*kh)%cg,oc=i/(kw*kh*cg),group=oc/og;
  for(int b=0;b<batch;++b)for(int oy=0;oy<oh;++oy)for(int ox=0;ox<ow;++ox){int iy=oy*sy-py+ky*dy,ix=ox*sx-px+kx*dx;if(iy>=0&&iy<h&&ix>=0&&ix<w)result+=X(0,((b*ci+group*cg+c)*h+iy)*w+ix)*X(2,((b*co+oc)*oh+oy)*ow+ox);}
}
return result;
