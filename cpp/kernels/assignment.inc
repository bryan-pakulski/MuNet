int queries=D[1],targets=D[2],b=i/targets,wanted=i%targets;
if(X(1,b*targets+wanted)<=0.0f)return -1.0f;
for(int t=0;t<=targets;++t)U[t]=0.0f;
for(int q=0;q<=queries;++q){V[q]=0.0f;P[q]=0;WAY[q]=0;}
for(int row=1;row<=targets;++row){
  if(X(1,b*targets+row-1)<=0.0f)continue;
  P[0]=row;int j0=0;
  for(int q=0;q<=queries;++q){MV[q]=1e30f;USED[q]=0;}
  for(int iteration=0;iteration<=queries;++iteration){
    USED[j0]=1;int i0=P[j0],j1=0;float delta=1e30f;
    for(int q=1;q<=queries;++q)if(USED[q]==0){float cost=X(0,(b*queries+q-1)*targets+i0-1);if(!isfinite(cost)||abs(cost)>1e20f)return -1.0f;float cur=cost-U[i0]-V[q];if(cur<MV[q]){MV[q]=cur;WAY[q]=j0;}if(MV[q]<delta){delta=MV[q];j1=q;}}
    if(j1==0)return -1.0f;
    for(int q=0;q<=queries;++q){if(USED[q]!=0){U[P[q]]+=delta;V[q]-=delta;}else MV[q]-=delta;}
    j0=j1;if(P[j0]==0)break;
  }
  for(int iteration=0;iteration<=queries;++iteration){int j1=WAY[j0];P[j0]=P[j1];j0=j1;if(j0==0)break;}
}
for(int q=1;q<=queries;++q)if(P[q]==wanted+1)return float(q-1);return -1.0f;
