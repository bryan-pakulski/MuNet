#include "ops.hpp"
#include "kernel_sources.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace munet {
namespace {
void check(bool ok,const std::string& s){if(!ok)throw std::invalid_argument(s);}
Shape bc(const Shape& a,const Shape& b){Shape o(std::max(a.size(),b.size()),1);for(size_t k=0;k<o.size();++k){auto x=k<a.size()?a[a.size()-1-k]:1,y=k<b.size()?b[b.size()-1-k]:1;check(x==y||x==1||y==1,"incompatible broadcast shapes");o[o.size()-1-k]=std::max(x,y);}return o;}
struct Spec{std::string file;int mode;};
Spec spec(const std::string& op){
  static const std::map<std::string,Spec> table={
    {"abs",{"pointwise",0}},{"sign",{"pointwise",1}},{"tanh",{"pointwise",2}},
    {"erf",{"pointwise",3}},{"gelu",{"pointwise",4}},{"gelu_grad",{"pointwise",5}},
    {"sin",{"pointwise",6}},{"cos",{"pointwise",7}},
    {"minimum",{"pointwise",8}},{"maximum",{"pointwise",9}},
    {"lt",{"pointwise",10}},{"le",{"pointwise",11}},{"eq",{"pointwise",12}},
    {"gt",{"pointwise",13}},{"ge",{"pointwise",14}},{"where",{"pointwise",15}},
    {"detach",{"pointwise",16}},{"softplus",{"pointwise",17}},{"floor",{"pointwise",18}},
    {"permute",{"permute",0}},{"slice",{"slice",0}},{"slice_grad",{"slice",1}},
    {"concat",{"concat",0}},{"matmul",{"matmul",0}},{"max",{"max",0}},
    {"conv2d",{"conv",0}},{"conv2d_dx",{"conv",1}},{"conv2d_dw",{"conv",2}},
    {"max_pool2d",{"pool",0}},{"avg_pool2d",{"pool",1}},
    {"max_pool2d_grad",{"pool",2}},{"avg_pool2d_grad",{"pool",3}},
    {"grid_sample",{"grid",0}},{"grid_sample_dx",{"grid",1}},{"grid_sample_dg",{"grid",2}},
    {"resize_nearest",{"resize",0}},{"resize_nearest_grad",{"resize",1}},
    {"gather",{"gather",0}},{"gather_grad",{"gather",1}},
    {"take",{"take",0}},{"take_grad",{"take",1}},
    {"topk_rank",{"topk",0}},{"topk_indices",{"topk",1}},
    {"assignment",{"assignment",0}},{"random_uniform",{"random",0}},
  };
  auto it=table.find(op);return it==table.end()?Spec{"",0}:it->second;
}
int axis(int64_t a,size_t r){if(a<0)a+=r;check(a>=0&&a<int64_t(r),"axis out of range");return int(a);}
Shape conv_shape(const Shape& x,const Shape& w,const Shape& a){
  check(x.size()==4&&w.size()==4&&a.size()==7,"Conv2d requires NCHW/OIHW and seven attributes");
  check(a[0]>0&&a[1]>0&&a[2]>=0&&a[3]>=0&&a[4]>0&&a[5]>0&&a[6]>0,"invalid convolution attributes");
  check(x[1]==w[1]*a[6]&&w[0]%a[6]==0,"convolution channels/groups mismatch");
  int64_t h=x[2]+2*a[2]-a[4]*(w[2]-1)-1,ww=x[3]+2*a[3]-a[5]*(w[3]-1)-1;
  check(h>=0&&ww>=0,"convolution kernel larger than padded input");return {x[0],w[0],h/a[0]+1,ww/a[1]+1};
}
Shape pool_shape(const Shape& x,const Shape& a){
  check(x.size()==4&&a.size()==8,"pooling requires NCHW and eight attributes");
  check(a[0]>0&&a[1]>0&&a[2]>0&&a[3]>0&&a[4]>=0&&a[5]>=0&&a[4]<=a[0]/2&&a[5]<=a[1]/2&&(a[6]==0||a[6]==1)&&(a[7]==0||a[7]==1),"invalid pooling attributes");
  Shape y=x;for(int d=0;d<2;++d){int64_t z=x[d+2]+2*a[d+4]-a[d]+(a[6]?a[d+2]-1:0);check(z>=0,"pooling kernel larger than input");y[d+2]=z/a[d+2]+1;if(a[6]&&(y[d+2]-1)*a[d+2]>=x[d+2]+a[d+4])--y[d+2];}return y;
}
float merf(float x){float z=std::abs(x),t=1.f/(1.f+0.3275911f*z);float p=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;return (x<0?-1.f:1.f)*(1.f-p*std::exp(-z*z));}
}
bool extended_shape(const Graph& g,const std::string& op,const std::vector<Id>& args,Shape& o,Shape& a){
  auto sp=spec(op);if(sp.file.empty())return false;
  check(!args.empty()&&args.size()<=16,"extended operation requires 1..16 inputs");
  auto s=[&](size_t k)->Shape{check(k<args.size(),"missing operation input");return g.at(args[k]).shape;};
  auto ar=[&](size_t n){check(args.size()==n,"wrong input count for "+op);};
  if(sp.file=="pointwise"){
    int mode=sp.mode;ar(mode==15?3:((mode>=8&&mode<=14)||mode==5)?2:1);check(a.empty(),"unexpected pointwise attributes");o=s(0);for(size_t k=1;k<args.size();++k)o=bc(o,s(k));
  }else if(op=="permute"){
    ar(1);check(a.size()==s(0).size(),"permutation rank mismatch");Shape seen=a;std::sort(seen.begin(),seen.end());for(size_t d=0;d<a.size();++d)check(seen[d]==int64_t(d),"invalid permutation");for(auto d:a)o.push_back(s(0)[d]);
  }else if(sp.file=="slice"){
    ar(sp.mode?2:1);auto x=s(sp.mode?1:0);size_t r=x.size();check(a.size()==r*3,"slice attribute rank mismatch");Shape sliced;for(size_t d=0;d<r;++d){check(a[r+d]>0&&a[2*r+d]!=0,"empty slices and zero steps are unsupported");auto end=a[d]+(a[r+d]-1)*a[2*r+d];check(a[d]>=0&&a[d]<x[d]&&end>=0&&end<x[d],"slice index out of bounds");sliced.push_back(a[r+d]);}if(sp.mode)check(s(0)==sliced,"slice adjoint shape mismatch");o=sp.mode?x:sliced;
  }else if(op=="concat"){
    check(a.size()==1,"concat needs an axis");o=s(0);a[0]=axis(a[0],o.size());o[a[0]]=0;for(size_t k=0;k<args.size();++k){auto x=s(k);check(x.size()==o.size(),"concat rank mismatch");for(size_t d=0;d<o.size();++d)if(d!=size_t(a[0]))check(x[d]==o[d],"concat dimension mismatch");o[a[0]]+=x[a[0]];}
  }else if(op=="max"){
    ar(1);o=s(0);for(auto& d:a)d=axis(d,o.size());std::sort(a.begin(),a.end());check(std::adjacent_find(a.begin(),a.end())==a.end(),"duplicate reduction axes");for(auto d:a)o[d]=1;
  }else if(sp.file=="conv"){
    ar(sp.mode?3:2);auto y=conv_shape(s(0),s(1),a);if(sp.mode)check(s(2)==y,"convolution adjoint shape mismatch");o=sp.mode==1?s(0):sp.mode==2?s(1):y;
  }else if(sp.file=="pool"){
    ar(sp.mode>=2?2:1);auto y=pool_shape(s(0),a);if(sp.mode>=2)check(s(1)==y,"pooling adjoint shape mismatch");o=sp.mode>=2?s(0):y;
  }else if(sp.file=="grid"){
    ar(sp.mode?3:2);check(a.empty(),"grid_sample implements bilinear/zeros/align_corners=False");auto x=s(0),grid=s(1);check(x.size()==4&&grid.size()==4&&grid[0]==x[0]&&grid[3]==2,"grid_sample requires NCHW and NHW2 grid");Shape y={x[0],x[1],grid[1],grid[2]};if(sp.mode)check(s(2)==y,"grid sampling adjoint shape mismatch");o=sp.mode==1?x:sp.mode==2?grid:y;
  }else if(sp.file=="resize"){
    ar(sp.mode?2:1);check(s(0).size()==4&&a.size()==2&&a[0]>0&&a[1]>0,"resize_nearest requires NCHW and positive output size");auto y=s(0);y[2]=a[0];y[3]=a[1];if(sp.mode)check(s(1)==y,"resize adjoint shape mismatch");o=sp.mode?s(0):y;
  }else if(sp.file=="gather"){
    ar(sp.mode?3:2);check(a.size()==1,"gather needs an axis");auto x=s(0),idx=s(1);a[0]=axis(a[0],x.size());check(x.size()==idx.size(),"gather index rank mismatch");for(size_t d=0;d<x.size();++d)if(d!=size_t(a[0]))check(x[d]==idx[d],"gather non-axis dimensions must match");check(x[a[0]]<=16777216,"index axis exceeds exact index representation");if(sp.mode)check(s(2)==idx,"gather adjoint shape mismatch");o=sp.mode?x:idx;
  }else if(sp.file=="take"){
    ar(sp.mode?3:2);check(a.size()==1,"take needs an axis");auto x=s(0),idx=s(1);a[0]=axis(a[0],x.size());check(x[a[0]]<=16777216,"index axis exceeds exact index representation");Shape y(x.begin(),x.begin()+a[0]);y.insert(y.end(),idx.begin(),idx.end());y.insert(y.end(),x.begin()+a[0]+1,x.end());if(sp.mode)check(s(2)==y,"take adjoint shape mismatch");o=sp.mode?x:y;
  }else if(sp.file=="topk"){
    ar(sp.mode?2:1);check(a.size()==2,"topk needs axis and k");o=s(0);a[0]=axis(a[0],o.size());check(a[1]>0&&a[1]<=o[a[0]]&&o[a[0]]<=16777216,"invalid topk size");if(sp.mode){check(s(1)==s(0),"topk rank shape mismatch");o[a[0]]=a[1];}
  }else if(op=="random_uniform"){
    ar(2);check(s(0).empty()&&s(1).empty(),"random seed and counter must be scalar");o=a;numel(o);
  }else if(op=="assignment"){
    ar(2);check(a.empty(),"assignment has no attributes");auto c=s(0),m=s(1);check(c.size()==3&&m==Shape{c[0],c[2]}&&c[2]<=c[1],"assignment requires cost[B,Q,T], valid[B,T], T<=Q");check(c[1]<=1024&&c[2]<=512,"assignment exceeds supported query/target bounds");o=m;
  }else return false;
  return true;
}
bool extended_gradient(Graph& g,Id id,Id dy,const std::function<void(Id,Id)>& acc){
  Node n=g.at(id);auto sp=spec(n.op);if(sp.file.empty())return false;
  auto op=[&](const std::string& k,std::vector<Id> x,Shape a=Shape{}){return g.op(k,x,a);};
  auto c=[&](float v){return g.leaf("constant","_ad_"+std::to_string(g.nodes.size()),{},{v});};
  Id x=n.inputs[0],y=n.inputs.size()>1?n.inputs[1]:-1;
  if(n.op=="detach"||n.op=="sign"||n.op=="floor"||(sp.file=="pointwise"&&sp.mode>=10&&sp.mode<=14)||sp.file=="topk"||n.op=="assignment"||n.op=="random_uniform")return true;
  if(n.op=="abs")acc(x,op("mul",{dy,op("sign",{x})}));
  else if(n.op=="tanh")acc(x,op("mul",{dy,op("sub",{c(1),op("mul",{id,id})})}));
  else if(n.op=="erf"){Id ex=op("exp",{op("neg",{op("mul",{x,x})})});acc(x,op("mul",{dy,op("mul",{c(1.1283791671f),ex})}));}
  else if(n.op=="gelu")acc(x,op("gelu_grad",{x,dy}));
  else if(n.op=="sin")acc(x,op("mul",{dy,op("cos",{x})}));
  else if(n.op=="cos")acc(x,op("neg",{op("mul",{dy,op("sin",{x})})}));
  else if(n.op=="softplus")acc(x,op("mul",{dy,op("sigmoid",{x})}));
  else if(n.op=="minimum"||n.op=="maximum"){
    Id tie=op("mul",{c(0.5f),op("eq",{x,y})});std::string cmp=n.op=="minimum"?"lt":"gt";
    acc(x,op("mul",{dy,op("add",{op(cmp,{x,y}),tie})}));acc(y,op("mul",{dy,op("add",{op(cmp,{y,x}),tie})}));
  }else if(n.op=="where"){acc(y,op("where",{x,dy,c(0)}));acc(n.inputs[2],op("where",{x,c(0),dy}));}
  else if(n.op=="permute"){Shape inv(n.attrs.size());for(size_t d=0;d<inv.size();++d)inv[n.attrs[d]]=d;acc(x,op("permute",{dy},inv));}
  else if(n.op=="slice")acc(x,op("slice_grad",{dy,x},n.attrs));
  else if(n.op=="concat"){int64_t start=0;for(auto v:n.inputs){auto sh=g.at(v).shape;Shape a(sh.size(),0);a[n.attrs[0]]=start;a.insert(a.end(),sh.begin(),sh.end());a.insert(a.end(),sh.size(),1);acc(v,op("slice",{dy},a));start+=sh[n.attrs[0]];}}
  else if(n.op=="max"){Id mask=op("eq",{x,op("broadcast",{id},g.at(x).shape)});Id norm=op("sum",{mask},n.attrs);acc(x,op("mul",{mask,op("div",{dy,norm})}));}
  else if(n.op=="conv2d"){acc(x,op("conv2d_dx",{x,y,dy},n.attrs));acc(y,op("conv2d_dw",{x,y,dy},n.attrs));}
  else if(n.op=="max_pool2d"||n.op=="avg_pool2d")acc(x,op(n.op+"_grad",{x,dy},n.attrs));
  else if(n.op=="grid_sample"){acc(x,op("grid_sample_dx",{x,y,dy}));acc(y,op("grid_sample_dg",{x,y,dy}));}
  else if(n.op=="resize_nearest")acc(x,op("resize_nearest_grad",{x,dy},n.attrs));
  else if(n.op=="gather"||n.op=="take")acc(x,op(n.op+"_grad",{x,y,dy},n.attrs));
  else return false;
  return true;
}
float Plan::extended_calculate(Id id,size_t index) const{
  const auto& n=graph.at(id);auto sp=spec(n.op);int MODE=sp.mode,i=int(index),RO=int(n.shape.size()),IN=int(n.inputs.size());
  std::array<int,8> O{};O.fill(1);for(int d=0;d<RO;++d)O[d]=int(n.shape[d]);
  std::array<int,128>D{};D.fill(1);std::array<int,16>R{},N{};std::array<int,32>A{};
  for(int a=0;a<IN;++a){const auto& s=graph.at(n.inputs[a]).shape;R[a]=int(s.size());N[a]=int(numel(s));for(int d=0;d<R[a];++d)D[a*8+d]=int(s[d]);}
  for(size_t a=0;a<n.attrs.size();++a)A[a]=int(n.attrs[a]);int NA=int(n.attrs.size());
  auto SD=[&](int a,int d){int v=1;for(int j=d+1;j<R[a];++j)v*=D[a*8+j];return v;};
  auto SO=[&](int d){int v=1;for(int j=d+1;j<RO;++j)v*=O[j];return v;};
  auto X=[&](int a,int j){return j<0||j>=N[a]?0.f:read_value(n.inputs[a],size_t(j));};
  auto BI=[&](int a,int v){int j=0;for(int d=0;d<R[a];++d)if(D[a*8+d]!=1)j+=(v/SO(d+RO-R[a])%O[d+RO-R[a]])*SD(a,d);return j;};
  using std::min;using std::max;using std::abs;using std::floor;using std::exp;using std::log;using std::tanh;using std::sin;using std::cos;using std::sqrt;using std::isfinite;
  using uint=uint32_t;
  if(sp.file=="random"){
#include "kernels/random.inc"
  }
  if(sp.file=="pointwise"){
#include "kernels/pointwise.inc"
  }if(sp.file=="permute"){
#include "kernels/permute.inc"
  }if(sp.file=="slice"){
#include "kernels/slice.inc"
  }if(sp.file=="concat"){
#include "kernels/concat.inc"
  }if(sp.file=="matmul"){
#include "kernels/matmul.inc"
  }if(sp.file=="max"){
#include "kernels/max.inc"
  }if(sp.file=="conv"){
#include "kernels/conv.inc"
  }if(sp.file=="pool"){
#include "kernels/pool.inc"
  }if(sp.file=="grid"){
#include "kernels/grid.inc"
  }if(sp.file=="resize"){
#include "kernels/resize.inc"
  }if(sp.file=="gather"){
#include "kernels/gather.inc"
  }if(sp.file=="take"){
#include "kernels/take.inc"
  }if(sp.file=="topk"){
#include "kernels/topk.inc"
  }if(sp.file=="assignment"){
    // The same portable solver body is compiled into C++ and GLSL.
    std::vector<float> U(D[2]+1),V(D[1]+1),MV(D[1]+1);std::vector<int>P(D[1]+1),WAY(D[1]+1),USED(D[1]+1);
#include "kernels/assignment.inc"
  }
  throw std::logic_error("missing extended kernel: "+n.op);
}
std::string Plan::extended_shader(Id id) const{
  const auto& n=graph.at(id);auto sp=spec(n.op);auto found=kernel_sources.find(sp.file);if(found==kernel_sources.end())throw std::logic_error("missing shader: "+n.op);
  std::ostringstream out;out<<shader_header(id);
  auto array=[&](const std::string& name,std::vector<int64_t> values,size_t count,int fill){values.resize(count,fill);out<<"const int "<<name<<"["<<count<<"]=int["<<count<<"](";for(size_t j=0;j<count;++j)out<<(j?",":"")<<values[j];out<<");\n";};
  array("O",n.shape,8,1);array("A",n.attrs,32,0);std::vector<int64_t>d(128,1),r(16,0),sz(16,0);for(size_t a=0;a<n.inputs.size();++a){auto s=graph.at(n.inputs[a]).shape;r[a]=s.size();sz[a]=numel(s);std::copy(s.begin(),s.end(),d.begin()+a*8);}array("D",d,128,1);array("R",r,16,0);array("N",sz,16,0);
  out<<"const int RO="<<n.shape.size()<<",IN="<<n.inputs.size()<<",NA="<<n.attrs.size()<<",MODE="<<sp.mode<<";\n";
  out<<"int SD(int a,int d){int v=1;for(int j=d+1;j<R[a];++j)v*=D[a*8+j];return v;}\nint SO(int d){int v=1;for(int j=d+1;j<RO;++j)v*=O[j];return v;}\nint BI(int a,int v){int j=0;for(int d=0;d<R[a];++d)if(D[a*8+d]!=1)j+=(v/SO(d+RO-R[a])%O[d+RO-R[a]])*SD(a,d);return j;}\n";
  out<<"bool isfinite(float x){return !isnan(x)&&!isinf(x);}\nfloat merf(float x){float z=abs(x),t=1.0/(1.0+0.3275911*z);float p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;return (x<0.0?-1.0:1.0)*(1.0-p*exp(-z*z));}\n";
  out<<"float X(int a,int j){if(j<0||j>=N[a])return 0.0;";for(size_t a=0;a<n.inputs.size();++a)out<<"if(a=="<<a<<")return "<<expr(n.inputs[a],"uint(j)")<<";";out<<"return 0.0;}\nfloat compute(int i){\n";
  if(sp.file=="assignment"){out<<"float U["<<d[2]+1<<"],V["<<d[1]+1<<"],MV["<<d[1]+1<<"];int P["<<d[1]+1<<"],WAY["<<d[1]+1<<"],USED["<<d[1]+1<<"];\n";}
  out<<found->second<<"\n}\nvoid main(){uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*gl_NumWorkGroups.x*64u;if(i>="<<numel(n.shape)<<"u)return;b"<<id<<".v[i]=compute(int(i));}\n";return out.str();
}
}
