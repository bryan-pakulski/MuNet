#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace munet {
namespace {
void require(bool ok,const std::string& message) { if(!ok) throw std::invalid_argument(message); }
bool is_leaf(const Node& n) { return n.op=="input"||n.op=="parameter"||n.op=="constant"; }
bool elementwise(const std::string& s) {
  return s=="add"||s=="sub"||s=="mul"||s=="div"||s=="neg"||s=="relu"||
    s=="relu_grad"||s=="sigmoid"||s=="exp"||s=="log"||s=="sqrt"||
    s=="reshape"||s=="broadcast"||s=="transpose"||s=="identity";
}
Shape broadcast_shape(const Shape& a,const Shape& b) {
  Shape out(std::max(a.size(),b.size()),1);
  for(size_t i=0;i<out.size();++i) {
    auto x=i<a.size()?a[a.size()-1-i]:1;
    auto y=i<b.size()?b[b.size()-1-i]:1;
    require(x==y||x==1||y==1,"incompatible broadcast shapes");
    out[out.size()-1-i]=std::max(x,y);
  }
  return out;
}
size_t stride(const Shape& s,size_t d) {
  size_t result=1; for(size_t i=d+1;i<s.size();++i) result*=s[i]; return result;
}
size_t broadcast_index(size_t i,const Shape& out,const Shape& in) {
  if(out==in) return i;
  size_t r=0,delta=out.size()-in.size();
  for(size_t d=0;d<in.size();++d) if(in[d]!=1)
    r+=((i/stride(out,d+delta))%out[d+delta])*stride(in,d);
  return r;
}
std::string uint_lit(size_t v) { return std::to_string(v)+"u"; }
std::string broadcast_expr(const std::string& i,const Shape& out,const Shape& in) {
  if(out==in) return i;
  std::string r="0u";size_t delta=out.size()-in.size();
  for(size_t d=0;d<in.size();++d) if(in[d]!=1)
    r+="+(("+i+")/"+uint_lit(stride(out,d+delta))+"%"+uint_lit(out[d+delta])+")*"+uint_lit(stride(in,d));
  return "("+r+")";
}
std::pair<size_t,size_t> reduction_index(size_t i,size_t k,const Node& n,const Shape& in) {
  size_t source=0,reduction_size=1;
  for(auto a:n.attrs) reduction_size*=in[a];
  size_t red_stride=reduction_size;
  for(size_t d=0;d<in.size();++d) {
    if(std::find(n.attrs.begin(),n.attrs.end(),d)!=n.attrs.end()) {
      red_stride/=in[d]; source+=((k/red_stride)%in[d])*stride(in,d);
    } else source+=((i/stride(n.shape,d))%n.shape[d])*stride(in,d);
  }
  return {source,reduction_size};
}
}
size_t numel(const Shape& shape) {
  require(shape.size()<=8,"rank above 8 is outside the v0 contract");
  size_t n=1;
  for(auto d:shape) {
    require(d>0,"v0 requires positive static dimensions");
    require(static_cast<uint64_t>(d)<=UINT32_MAX/4/n,"tensor exceeds v0 32-bit indexing limit");
    n*=d;
  }
  return n;
}
const Node& Graph::at(Id id) const {
  require(id>=0&&static_cast<size_t>(id)<nodes.size(),"invalid value id"); return nodes[id];
}
Id Graph::leaf(const std::string& kind,const std::string& name,const Shape& shape,const std::vector<float>& data) {
  require(kind=="input"||kind=="parameter"||kind=="constant","invalid leaf kind");
  require(!name.empty(),"leaf name must not be empty");
  require(std::none_of(nodes.begin(),nodes.end(),[&](const Node& n){return is_leaf(n)&&n.name==name;}),"duplicate leaf name: "+name);
  auto count=numel(shape);
  require(kind=="input"?data.empty():data.size()==count,"leaf data size does not match shape");
  nodes.push_back({kind,name,{},shape,{},data});return static_cast<Id>(nodes.size()-1);
}
Id Graph::op(const std::string& kind,const std::vector<Id>& args,const Shape& attrs) {
  for(auto id:args) at(id);
  auto arity=[&](size_t n){require(args.size()==n,"wrong number of inputs for "+kind);};
  Shape shape,normalized=attrs;
  if(kind=="add"||kind=="sub"||kind=="mul"||kind=="div"||kind=="relu_grad") {
    arity(2);shape=broadcast_shape(at(args[0]).shape,at(args[1]).shape);
  } else if(kind=="matmul") {
    arity(2);auto a=at(args[0]).shape,b=at(args[1]).shape;
    require(a.size()==2&&b.size()==2&&a[1]==b[0],"matmul requires compatible rank-2 tensors");shape={a[0],b[1]};
  } else if(kind=="transpose") {
    arity(1);auto a=at(args[0]).shape;require(a.size()==2,"transpose requires rank 2");shape={a[1],a[0]};
  } else if(kind=="reshape") {
    arity(1);require(numel(attrs)==numel(at(args[0]).shape),"reshape changes element count");shape=attrs;
  } else if(kind=="broadcast") {
    arity(1);require(broadcast_shape(at(args[0]).shape,attrs)==attrs,"invalid broadcast target");shape=attrs;
  } else if(kind=="sum") {
    arity(1);shape=at(args[0]).shape;
    for(auto& a:normalized) {
      if(a<0) a+=shape.size();
      require(a>=0&&static_cast<size_t>(a)<shape.size(),"reduction axis out of range");
    }
    std::sort(normalized.begin(),normalized.end());
    require(std::adjacent_find(normalized.begin(),normalized.end())==normalized.end(),"duplicate reduction axes");
    for(auto a:normalized) shape[a]=1;
  } else if(kind=="neg"||kind=="relu"||kind=="sigmoid"||kind=="exp"||kind=="log"||kind=="sqrt"||kind=="identity") {
    arity(1);shape=at(args[0]).shape;
  } else throw std::invalid_argument("unsupported native op: "+kind);
  if(kind!="sum"&&kind!="reshape"&&kind!="broadcast") require(attrs.empty(),"unexpected attributes for "+kind);
  numel(shape);nodes.push_back({kind,"",args,shape,normalized,{}});return static_cast<Id>(nodes.size()-1);
}
Id Graph::scalar(float v) {return leaf("constant","_scalar_"+std::to_string(nodes.size()),{},{v});}
Id Graph::sum_to(Id value,const Shape& target_ref) {
  const Shape target=target_ref;
  auto source=at(value).shape;
  require(broadcast_shape(source,target)==source,"gradient shape cannot be reduced to input");
  Shape axes;size_t delta=source.size()-target.size();
  for(size_t d=0;d<source.size();++d)
    if(d<delta||(target[d-delta]==1&&source[d]!=1)) axes.push_back(d);
  if(!axes.empty()) value=op("sum",{value},axes);
  if(at(value).shape!=target) value=op("reshape",{value},target);
  return value;
}
std::vector<Id> Graph::gradients(Id loss,const std::vector<Id>& wrt) {
  require(numel(at(loss).shape)==1,"backward requires a scalar loss");
  size_t end=nodes.size();for(auto id:wrt) at(id);
  std::vector<Id> grads(end,-1);Id one=scalar(1.f);grads[loss]=op("broadcast",{one},at(loss).shape);
  auto acc=[&](Id target,Id value) {
    value=sum_to(value,at(target).shape);
    grads[target]=grads[target]<0?value:op("add",{grads[target],value});
  };
  for(Id id=static_cast<Id>(end)-1;id>=0;--id) {
    if(grads[id]<0) continue;
    Node n=nodes[id]; // Appending derivative nodes may reallocate the vector.
    if(is_leaf(n)) continue;
    Id g=grads[id],a=n.inputs[0],b=n.inputs.size()>1?n.inputs[1]:-1;
    if(n.op=="add") {acc(a,g);acc(b,g);}
    else if(n.op=="sub") {acc(a,g);acc(b,op("neg",{g}));}
    else if(n.op=="mul") {acc(a,op("mul",{g,b}));acc(b,op("mul",{g,a}));}
    else if(n.op=="div") {acc(a,op("div",{g,b}));acc(b,op("neg",{op("div",{op("mul",{g,a}),op("mul",{b,b})})}));}
    else if(n.op=="neg") acc(a,op("neg",{g}));
    else if(n.op=="matmul") {acc(a,op("matmul",{g,op("transpose",{b})}));acc(b,op("matmul",{op("transpose",{a}),g}));}
    else if(n.op=="transpose") acc(a,op("transpose",{g}));
    else if(n.op=="reshape") acc(a,op("reshape",{g,},at(a).shape));
    else if(n.op=="broadcast"||n.op=="identity") acc(a,g);
    else if(n.op=="sum") acc(a,op("broadcast",{g},at(a).shape));
    else if(n.op=="relu") acc(a,op("relu_grad",{a,g}));
    else if(n.op=="sigmoid") acc(a,op("mul",{g,op("mul",{id,op("sub",{scalar(1.f),id})})}));
    else if(n.op=="exp") acc(a,op("mul",{g,id}));
    else if(n.op=="log") acc(a,op("div",{g,a}));
    else if(n.op=="sqrt") acc(a,op("div",{g,op("mul",{scalar(2.f),id})}));
    else throw std::invalid_argument("no derivative rule for "+n.op+" (higher-order autodiff is not implemented)");
  }
  std::vector<Id> result;
  for(auto id:wrt) {if(grads[id]<0){Id zero=scalar(0.f);result.push_back(op("broadcast",{zero},at(id).shape));}else result.push_back(grads[id]);}
  return result;
}

Plan::Plan(const Graph& g,const std::vector<Id>& result,const std::vector<std::pair<Id,Id>>& writes,bool fuse)
  :graph(g),outputs(result),updates(writes) {
  require(!outputs.empty(),"a plan needs at least one output");
  std::set<Id> roots(outputs.begin(),outputs.end()),destinations;
  for(auto& [dst,src]:updates) {
    require(graph.at(dst).op=="parameter","updates must target parameters");
    require(graph.at(dst).shape==graph.at(src).shape,"update shape mismatch");
    require(destinations.insert(dst).second,"duplicate parameter update");
    // Snapshot all update values before any parameter is overwritten (including swaps).
    if(graph.at(src).op!="identity") src=graph.op("identity",{src});roots.insert(src);
  }
  size_t n=graph.nodes.size();live_.assign(n,false);inline_.assign(n,false);
  std::function<void(Id)> visit=[&](Id id){graph.at(id);if(live_[id])return;live_[id]=true;for(auto x:graph.at(id).inputs)visit(x);};
  for(auto id:roots) visit(id);for(auto [dst,src]:updates) visit(dst);
  std::vector<size_t> users(n,0);
  for(size_t i=0;i<n;++i) if(live_[i]) for(auto x:graph.nodes[i].inputs) ++users[x];
  for(size_t i=0;i<n;++i) if(live_[i]) {
    naive_floats_+=numel(graph.nodes[i].shape);
    if(!is_leaf(graph.nodes[i])) {
      ++unfused_nodes_;
      inline_[i]=fuse&&users[i]==1&&!roots.count(i)&&elementwise(graph.nodes[i].op);
    }
  }
  for(size_t i=0;i<n;++i) if(live_[i]&&!is_leaf(graph.nodes[i])&&!inline_[i])
    kernels.push_back({static_cast<Id>(i),numel(graph.nodes[i].shape),""});
  std::vector<int> last(n,-1);
  std::function<void(Id,int)> used=[&](Id id,int k) {
    if(inline_[id]) for(auto x:graph.at(id).inputs) used(x,k);
    else last[id]=std::max(last[id],k);
  };
  for(size_t k=0;k<kernels.size();++k) for(auto x:graph.at(kernels[k].root).inputs) used(x,k);
  for(auto id:roots) last[id]=kernels.size();
  offsets_.assign(n,std::numeric_limits<size_t>::max());
  size_t total=0;
  auto allocate=[&](size_t count){size_t off=total;total+=((count+63)/64)*64;return off;};
  for(size_t i=0;i<n;++i) if(live_[i]&&is_leaf(graph.nodes[i])) {
    offsets_[i]=allocate(numel(graph.nodes[i].shape));
    if(graph.nodes[i].op=="input") inputs.push_back(i);
  }
  struct Slot {size_t offset,count;int last;};std::vector<Slot> slots;
  for(size_t k=0;k<kernels.size();++k) {
    auto id=kernels[k].root;auto size=numel(graph.at(id).shape);
    auto found=slots.end();
    for(auto it=slots.begin();it!=slots.end();++it) if(it->last<static_cast<int>(k)&&it->count>=size&&(found==slots.end()||it->count<found->count)) found=it;
    if(found==slots.end()) {auto off=allocate(size);slots.push_back({off,((size+63)/64)*64,last[id]});offsets_[id]=off;}
    else {offsets_[id]=found->offset;found->last=last[id];}
  }
  require(total<=UINT32_MAX/4,"arena exceeds v0 32-bit indexing limit");
  arena_.resize(total,0.f);arena_floats_=total;
  for(size_t i=0;i<n;++i) if(live_[i]&&is_leaf(graph.nodes[i])&&!graph.nodes[i].data.empty())
    std::copy(graph.nodes[i].data.begin(),graph.nodes[i].data.end(),arena_.begin()+offsets_[i]);
  for(auto& kernel:kernels) kernel.source=shader(kernel.root);
}
Plan::~Plan()=default;
float Plan::read_value(Id id,size_t index) const {
  return inline_[id]?calculate(id,index):arena_[offsets_[id]+index];
}
float Plan::calculate(Id id,size_t i) const {
  const auto& n=graph.at(id);
  auto get=[&](size_t arg){auto x=n.inputs[arg];return read_value(x,broadcast_index(i,n.shape,graph.at(x).shape));};
  if(n.op=="add") return get(0)+get(1);
  if(n.op=="sub") return get(0)-get(1);
  if(n.op=="mul") return get(0)*get(1);
  if(n.op=="div") return get(0)/get(1);
  if(n.op=="neg") return -get(0);
  if(n.op=="relu") return std::max(get(0),0.f);
  if(n.op=="relu_grad") return get(0)>0.f?get(1):0.f;
  if(n.op=="sigmoid") {auto x=get(0),e=std::exp(-std::abs(x));return x>=0?1.f/(1.f+e):e/(1.f+e);}
  if(n.op=="exp") return std::exp(get(0));
  if(n.op=="log") return std::log(get(0));
  if(n.op=="sqrt") return std::sqrt(get(0));
  if(n.op=="identity"||n.op=="reshape") return read_value(n.inputs[0],i);
  if(n.op=="broadcast") return get(0);
  if(n.op=="transpose") return read_value(n.inputs[0],(i%n.shape[1])*n.shape[0]+i/n.shape[1]);
  if(n.op=="matmul") {
    auto a=n.inputs[0],b=n.inputs[1];size_t K=graph.at(a).shape[1],N=n.shape[1];float v=0;
    for(size_t k=0;k<K;++k) v+=read_value(a,(i/N)*K+k)*read_value(b,k*N+i%N);
    return v;
  }
  if(n.op=="sum") {
    auto a=n.inputs[0];float v=0;auto count=reduction_index(i,0,n,graph.at(a).shape).second;
    for(size_t k=0;k<count;++k) v+=read_value(a,reduction_index(i,k,n,graph.at(a).shape).first);
    return v;
  }
  throw std::logic_error("missing CPU implementation: "+n.op);
}
std::string Plan::expr(Id id,const std::string& i,bool force) const {
  if(!force&&!inline_[id]) return "buf.v["+uint_lit(offsets_[id])+"+("+i+")]";
  const auto& n=graph.at(id);
  auto get=[&](size_t arg){auto x=n.inputs[arg];return expr(x,broadcast_expr(i,n.shape,graph.at(x).shape));};
  if(n.op=="add"||n.op=="sub"||n.op=="mul"||n.op=="div") {
    std::string token=n.op=="add"?"+":n.op=="sub"?"-":n.op=="mul"?"*":"/";
    return "("+get(0)+token+get(1)+")";
  }
  if(n.op=="neg") return "(-"+get(0)+")";
  if(n.op=="relu") return "max("+get(0)+",0.0)";
  if(n.op=="relu_grad") return "("+get(0)+">0.0?"+get(1)+":0.0)";
  if(n.op=="sigmoid") return "(1.0/(1.0+exp(-"+get(0)+")))";
  if(n.op=="exp"||n.op=="log"||n.op=="sqrt") return n.op+"("+get(0)+")";
  if(n.op=="reshape"||n.op=="identity") return expr(n.inputs[0],i);
  if(n.op=="broadcast") return get(0);
  if(n.op=="transpose") return expr(n.inputs[0],"(("+i+")%"+uint_lit(n.shape[1])+"*"+uint_lit(n.shape[0])+"+("+i+")/"+uint_lit(n.shape[1])+")");
  throw std::logic_error("op is not a scalar expression: "+n.op);
}
std::string Plan::shader(Id id) const {
  const auto& n=graph.at(id);
  std::ostringstream out;
  out<<"#version 450\nlayout(local_size_x=64) in;\nlayout(set=0,binding=0,std430) buffer Arena {float v[];} buf;\nvoid main(){uint i=gl_GlobalInvocationID.x;if(i>="<<uint_lit(numel(n.shape))<<")return;\n";
  if(n.op=="matmul") {
    auto a=n.inputs[0],b=n.inputs[1];auto K=graph.at(a).shape[1],N=n.shape[1];
    out<<"float r=0.0;for(uint k=0u;k<"<<uint_lit(K)<<";++k)r+="
       <<expr(a,"(i/"+uint_lit(N)+"*"+uint_lit(K)+"+k)")<<"*"
       <<expr(b,"(k*"+uint_lit(N)+"+i%"+uint_lit(N)+")")<<";\n";
  } else if(n.op=="sum") {
    auto a=n.inputs[0];const auto& in=graph.at(a).shape;size_t count=1;
    for(auto ax:n.attrs) count*=in[ax];size_t red_stride=count;
    std::string index="0u";
    for(size_t d=0;d<in.size();++d) {
      if(std::find(n.attrs.begin(),n.attrs.end(),d)!=n.attrs.end()) {
        red_stride/=in[d];index+="+(k/"+uint_lit(red_stride)+"%"+uint_lit(in[d])+")*"+uint_lit(stride(in,d));
      } else index+="+(i/"+uint_lit(stride(n.shape,d))+"%"+uint_lit(n.shape[d])+")*"+uint_lit(stride(in,d));
    }
    out<<"float r=0.0;for(uint k=0u;k<"<<uint_lit(count)<<";++k)r+="<<expr(a,"("+index+")")<<";\n";
  } else out<<"float r="<<expr(id,"i",true)<<";\n";
  out<<"buf.v["<<uint_lit(offsets_[id])<<"+i]=r;}\n";return out.str();
}
void Plan::enable_vulkan(const std::vector<std::vector<uint32_t>>& spirv,unsigned index) {
  require(counters_.runs==0&&!device_,"select the device before first execution");
  std::vector<std::pair<size_t,size_t>> ranges;
  for(auto id:inputs) ranges.push_back({offsets_[id],numel(graph.at(id).shape)});
  std::vector<std::pair<size_t,std::pair<size_t,size_t>>> copies;
  for(auto [dst,src]:updates) copies.push_back({offsets_[dst],{offsets_[src],numel(graph.at(dst).shape)}});
  device_=make_vulkan(arena_,kernels,spirv,ranges,copies,index,counters_);
  arena_.clear();arena_.shrink_to_fit();
}
void Plan::run(const std::vector<std::vector<float>>& feeds) {
  require(feeds.size()==inputs.size(),"wrong number of runtime inputs");
  std::vector<std::pair<size_t,std::vector<float>>> uploads;
  for(size_t i=0;i<inputs.size();++i) {
    auto id=inputs[i];require(feeds[i].size()==numel(graph.at(id).shape),"input element count mismatch");
    if(device_) uploads.push_back({offsets_[id],feeds[i]});
    else std::copy(feeds[i].begin(),feeds[i].end(),arena_.begin()+offsets_[id]);
  }
  if(device_) device_->run(uploads);
  else {
    for(auto& k:kernels) for(size_t i=0;i<k.count;++i) arena_[offsets_[k.root]+i]=calculate(k.root,i);
    for(auto [dst,src]:updates) std::copy_n(arena_.begin()+offsets_[src],numel(graph.at(dst).shape),arena_.begin()+offsets_[dst]);
  }
  ++counters_.runs;counters_.dispatches+=kernels.size();
}
std::vector<float> Plan::read(Id id) {
  graph.at(id);if(!live_[id]&&(graph.at(id).op=="parameter"||graph.at(id).op=="constant"))return graph.at(id).data;
  require(live_[id]&&!inline_[id],"value was removed or fused; expose it as an output to inspect it");
  require(is_leaf(graph.at(id))||std::find(outputs.begin(),outputs.end(),id)!=outputs.end(),"intermediate storage may have been reused; expose it as an output");
  auto size=numel(graph.at(id).shape),off=offsets_[id];
  if(device_) return device_->read(off,size);
  return {arena_.begin()+off,arena_.begin()+off+size};
}
void Plan::write(Id id,const std::vector<float>& data) {
  require(graph.at(id).op=="parameter","write requires a parameter");
  require(data.size()==numel(graph.at(id).shape),"parameter element count mismatch");
  if(!live_[id]){graph.nodes[id].data=data;return;}
  if(device_) device_->write(offsets_[id],data);
  else std::copy(data.begin(),data.end(),arena_.begin()+offsets_[id]);
}
void Plan::synchronize(){if(device_)device_->synchronize();}
std::string Plan::device_name() const{return device_?device_->name():"CPU reference";}
std::map<std::string,uint64_t> Plan::stats() const {
  return {{"runs",counters_.runs},{"submissions",counters_.submissions},{"dispatches",counters_.dispatches},
    {"upload_bytes",counters_.upload_bytes},{"download_bytes",counters_.download_bytes},{"waits",counters_.waits},
    {"kernels",kernels.size()},{"unfused_nodes",unfused_nodes_},{"arena_bytes",arena_floats_*sizeof(float)},
    {"naive_value_bytes",naive_floats_*sizeof(float)}};
}
}
