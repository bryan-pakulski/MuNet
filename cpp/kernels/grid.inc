int channels=D[1],h=D[2],w=D[3],gh=D[9],gw=D[10],points=gh*gw;
float result=0.0f;
if(MODE==1){
  int ix=i%w,iy=i/w%h,c=i/(w*h)%channels,b=i/(w*h*channels);
  for(int p=0;p<points;++p){float gx=(X(1,(b*points+p)*2)+1.0f)*float(w)*0.5f-0.5f,gy=(X(1,(b*points+p)*2+1)+1.0f)*float(h)*0.5f-0.5f;float wx=1.0f-abs(gx-float(ix)),wy=1.0f-abs(gy-float(iy));if(wx>0.0f&&wy>0.0f)result+=X(2,(b*channels+c)*points+p)*wx*wy;}
  return result;
}
int b=MODE==0?i/(channels*points):i/(points*2),p=MODE==0?i%points:i/2%points;
float gx=(X(1,(b*points+p)*2)+1.0f)*float(w)*0.5f-0.5f,gy=(X(1,(b*points+p)*2+1)+1.0f)*float(h)*0.5f-0.5f;
if(!isfinite(gx)||!isfinite(gy)||gx<-1.0f||gx>float(w)||gy<-1.0f||gy>float(h))return 0.0f;
int x0=int(floor(gx)),y0=int(floor(gy));float fx=gx-float(x0),fy=gy-float(y0);
int clo=MODE==0?i/points%channels:0,chi=MODE==0?clo+1:channels;
for(int c=clo;c<chi;++c){float value=0.0f;for(int ky=0;ky<2;++ky)for(int kx=0;kx<2;++kx){int x=x0+kx,y=y0+ky;if(x<0||x>=w||y<0||y>=h)continue;float wx=kx==0?1.0f-fx:fx,wy=ky==0?1.0f-fy:fy;if(MODE==2){if(i%2==0)wx=(kx==0?-1.0f:1.0f)*float(w)*0.5f;else wy=(ky==0?-1.0f:1.0f)*float(h)*0.5f;}value+=X(0,((b*channels+c)*h+y)*w+x)*wx*wy;}result+=MODE==0?value:value*X(2,(b*channels+c)*points+p);}
return result;
