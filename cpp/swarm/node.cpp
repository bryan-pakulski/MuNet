// Native numerical worker. No Python, Python modules, or shader compiler required.
#include "core.hpp"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace fs=std::filesystem;
using json=nlohmann::json;
using namespace munet;
namespace {
constexpr const char* ABI="munet-swarm-1";
constexpr size_t MAX_ARTIFACT=256ull*1024*1024, MAX_RESPONSE=32ull*1024*1024;
volatile std::sig_atomic_t stopping=0;
void stop(int){stopping=1;}
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
std::string sha(const std::string& data){
  unsigned char bytes[EVP_MAX_MD_SIZE];unsigned int count=0;
  require(EVP_Digest(data.data(),data.size(),bytes,&count,EVP_sha256(),nullptr)==1,"SHA256 failed");
  std::string out;const char* hex="0123456789abcdef";
  for(unsigned i=0;i<count;++i){out+=hex[bytes[i]>>4];out+=hex[bytes[i]&15];}return out;
}
std::string random_id(){
  unsigned char bytes[16];require(RAND_bytes(bytes,sizeof(bytes))==1,"secure random source failed");
  return sha(std::string(reinterpret_cast<char*>(bytes),sizeof(bytes))).substr(0,32);
}
bool valid_hash(const std::string& s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
std::string read_file(const fs::path& p,size_t limit=MAX_ARTIFACT){
  require(fs::file_size(p)<=limit,"file exceeds prototype limit");
  std::ifstream f(p,std::ios::binary);require(bool(f),"cannot open local file");
  std::string data((std::istreambuf_iterator<char>(f)),{});require(!f.bad(),"local read failed");return data;
}
void sync_directory(const fs::path& p){
  int fd=::open(p.c_str(),O_RDONLY|O_DIRECTORY);require(fd>=0,"cannot open state directory");
  int result=::fsync(fd);::close(fd);require(result==0,"directory fsync failed");
}
void atomic_write(const fs::path& p,const std::string& data){
  fs::create_directories(p.parent_path());auto tmp=p.string()+".tmp";
  int fd=::open(tmp.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);require(fd>=0,"cannot write local state");
  size_t done=0;
  while(done<data.size()){
    auto n=::write(fd,data.data()+done,data.size()-done);
    if(n<=0){::close(fd);throw std::runtime_error("state write failed");}done+=size_t(n);
  }
  int result=::fsync(fd);::close(fd);require(result==0,"state fsync failed");
  fs::rename(tmp,p);sync_directory(p.parent_path());
}
void remove_durable(const fs::path& p){if(fs::remove(p))sync_directory(p.parent_path());}
struct Lock {
  int fd=-1;
  explicit Lock(const fs::path& p){
    fd=::open(p.c_str(),O_CREAT|O_RDWR,0600);require(fd>=0,"cannot open node lock");
    if(flock(fd,LOCK_EX|LOCK_NB)!=0){::close(fd);fd=-1;throw std::runtime_error("another node uses this state directory");}
  }
  ~Lock(){if(fd>=0)::close(fd);}
};
struct HttpError:std::runtime_error {long status;HttpError(long s,const std::string& m):std::runtime_error(m),status(s){}};
struct Sink {std::string data;size_t limit=MAX_RESPONSE;};
size_t append(char* p,size_t a,size_t b,void* opaque){
  auto& s=*static_cast<Sink*>(opaque);size_t n=a*b;
  if(n>s.limit-s.data.size())return 0;
  try{s.data.append(p,n);}catch(...){return 0;}return n;
}
struct FileSink {FILE* file;size_t bytes;};
size_t append_file(char* p,size_t a,size_t b,void* opaque){
  auto& s=*static_cast<FileSink*>(opaque);size_t n=a*b;
  if(n>MAX_ARTIFACT-s.bytes)return 0;
  size_t written=fwrite(p,1,n,s.file);s.bytes+=written;return written;
}
struct Headers {curl_slist* p=nullptr;~Headers(){curl_slist_free_all(p);}
  void add(const std::string& s){auto* next=curl_slist_append(p,s.c_str());require(next,"HTTP header allocation failed");p=next;}};
struct Curl {CURL* p=curl_easy_init();Curl(){require(p,"curl initialization failed");}~Curl(){curl_easy_cleanup(p);}};
struct Url {CURLU* p=curl_url();Url(){require(p,"URL allocation failed");}~Url(){curl_url_cleanup(p);}};
std::string url_part(CURLU* p,CURLUPart part){char* value=nullptr;auto result=curl_url_get(p,part,&value,0);
  if(result!=CURLUE_OK)return {};std::string s(value);curl_free(value);return s;}
std::string validate_owner(std::string owner,bool allow_http){
  Url u;require(curl_url_set(u.p,CURLUPART_URL,owner.c_str(),0)==CURLUE_OK,"invalid owner URL");
  auto scheme=url_part(u.p,CURLUPART_SCHEME),host=url_part(u.p,CURLUPART_HOST),path=url_part(u.p,CURLUPART_PATH);
  require((scheme=="http"||scheme=="https")&&!host.empty(),"owner must use HTTP(S)");
  require(url_part(u.p,CURLUPART_USER).empty()&&url_part(u.p,CURLUPART_PASSWORD).empty()&&
          url_part(u.p,CURLUPART_QUERY).empty()&&url_part(u.p,CURLUPART_FRAGMENT).empty()&&
          (path.empty()||path=="/"),"owner URL must be an origin without credentials/path/query");
  require(scheme=="https"||host=="127.0.0.1"||host=="localhost"||host=="[::1]"||allow_http,
          "plaintext remote owner requires explicit --allow-http on a trusted private network");
  while(owner.back()=='/')owner.pop_back();return owner;
}
class Client {
  std::string owner_,token_;
  void configure(CURL* c,const std::string& path,Headers& h){
    std::string url=owner_+path;
    curl_easy_setopt(c,CURLOPT_URL,url.c_str());
    curl_easy_setopt(c,CURLOPT_NOSIGNAL,1L);
    curl_easy_setopt(c,CURLOPT_CONNECTTIMEOUT,3L);
    curl_easy_setopt(c,CURLOPT_TIMEOUT,30L);
    curl_easy_setopt(c,CURLOPT_LOW_SPEED_LIMIT,128L);
    curl_easy_setopt(c,CURLOPT_LOW_SPEED_TIME,15L);
    curl_easy_setopt(c,CURLOPT_FOLLOWLOCATION,0L);
    curl_easy_setopt(c,CURLOPT_SSL_VERIFYPEER,1L);
    curl_easy_setopt(c,CURLOPT_SSL_VERIFYHOST,2L);
    // Repaired wheels can contain a libcurl built on another Linux distribution.
    // Select this host's trust store instead of its build-time certificate path.
    if(const char* ca=std::getenv("MUNET_CA_BUNDLE")) {
      require(fs::is_regular_file(ca),"MUNET_CA_BUNDLE is not a certificate file");
      curl_easy_setopt(c,CURLOPT_CAINFO,ca);
    } else {
      for(const char* ca:{"/etc/ssl/certs/ca-certificates.crt","/etc/pki/tls/certs/ca-bundle.crt","/etc/ssl/cert.pem"})
        if(fs::is_regular_file(ca)){curl_easy_setopt(c,CURLOPT_CAINFO,ca);break;}
    }
    h.add("Authorization: Bearer "+token_);h.add("Content-Type: application/json");h.add("Expect:");
    curl_easy_setopt(c,CURLOPT_HTTPHEADER,h.p);
  }
 public:
  Client(std::string owner,std::string token):owner_(std::move(owner)),token_(std::move(token)){}
  json post(const std::string& path,const json& message){
    Curl c;Headers h;configure(c.p,path,h);auto body=message.dump();
    require(body.size()<=MAX_RESPONSE,"result exceeds 32 MiB protocol limit");
    Sink sink;curl_easy_setopt(c.p,CURLOPT_POST,1L);curl_easy_setopt(c.p,CURLOPT_POSTFIELDS,body.data());
    curl_easy_setopt(c.p,CURLOPT_POSTFIELDSIZE_LARGE,curl_off_t(body.size()));
    curl_easy_setopt(c.p,CURLOPT_WRITEFUNCTION,append);curl_easy_setopt(c.p,CURLOPT_WRITEDATA,&sink);
    auto result=curl_easy_perform(c.p);require(result==CURLE_OK,std::string("owner unavailable: ")+curl_easy_strerror(result));
    long status=0;curl_easy_getinfo(c.p,CURLINFO_RESPONSE_CODE,&status);
    if(status!=200)throw HttpError(status,"owner HTTP "+std::to_string(status)+" on "+path);
    return json::parse(sink.data);
  }
  void download(const std::string& key,const fs::path& directory){
    require(valid_hash(key),"invalid artifact hash");auto path=directory/key;
    if(fs::exists(path)){
      if(sha(read_file(path))==key)return;
      remove_durable(path);
    }
    auto part=fs::path(path.string()+".part");size_t offset=fs::exists(part)?fs::file_size(part):0;
    if(offset>MAX_ARTIFACT){remove_durable(part);offset=0;}
    if(offset&&sha(read_file(part))==key){fs::rename(part,path);sync_directory(directory);return;}
    Curl c;Headers h;configure(c.p,"/v1/artifacts/"+key,h);
    FILE* f=fopen(part.c_str(),"ab");require(f,"cannot write artifact cache");
    FileSink sink{f,offset};curl_easy_setopt(c.p,CURLOPT_WRITEFUNCTION,append_file);curl_easy_setopt(c.p,CURLOPT_WRITEDATA,&sink);
    if(offset)curl_easy_setopt(c.p,CURLOPT_RESUME_FROM_LARGE,curl_off_t(offset));
    auto result=curl_easy_perform(c.p);long status=0;curl_easy_getinfo(c.p,CURLINFO_RESPONSE_CODE,&status);
    bool synced=fflush(f)==0&&fsync(fileno(f))==0;fclose(f);require(synced,"artifact fsync failed");
    if(status!=0&&status!=200&&status!=206){remove_durable(part);throw HttpError(status,"artifact download HTTP "+std::to_string(status));}
    require(result==CURLE_OK,std::string("artifact download interrupted: ")+curl_easy_strerror(result));
    if(sha(read_file(part))!=key){remove_durable(part);throw std::runtime_error("artifact checksum mismatch");}
    fs::rename(part,path);sync_directory(directory);
  }
};

struct Options {
  std::string owner,device="vulkan:0",state="munet-node-state",name="node";
  uint64_t memory=0,max_batch=1024;double rate=0,poll=2;bool allow_http=false,once=false;
};
Options options(int argc,char** argv){
  Options o;
  for(int i=1;i<argc;++i){
    std::string a=argv[i];auto value=[&](){require(i+1<argc,"missing option value for "+a);return std::string(argv[++i]);};
    if(a=="--owner")o.owner=value();else if(a=="--device")o.device=value();
    else if(a=="--state")o.state=value();else if(a=="--name")o.name=value();
    else if(a=="--memory-mib")o.memory=std::stoull(value())*1024*1024;
    else if(a=="--max-batch")o.max_batch=std::stoull(value());
    else if(a=="--samples-per-second")o.rate=std::stod(value());
    else if(a=="--poll-seconds")o.poll=std::stod(value());
    else if(a=="--allow-http")o.allow_http=true;else if(a=="--once")o.once=true;
    else if(a=="--version"){std::cout<<"MuNet node "<<MUNET_VERSION<<std::endl;std::exit(0);}
    else if(a=="--help"){
      std::cout<<"munet-node --owner URL [--device vulkan:N|cpu] [--state DIR] [--name NAME]\n"
                 "  [--memory-mib N] [--max-batch N] [--samples-per-second N]\n"
                 "  [--poll-seconds N] [--allow-http] [--once]\n"
                 "Set MUNET_SWARM_TOKEN in the environment. --once processes one bundle;\n"
                 "exit 2 means connection or durable work/results need another attempt.\n";std::exit(0);
    }else throw std::runtime_error("unknown option: "+a);
  }
  require(!o.owner.empty(),"--owner is required");
  require(o.max_batch>0&&o.max_batch<=1'000'000,"invalid maximum batch");
  require(std::isfinite(o.rate)&&o.rate>=0&&o.rate<=1e12,"invalid throughput estimate");
  require(std::isfinite(o.poll)&&o.poll>=0.01&&o.poll<=60,"poll interval must be 0.01..60 seconds");
  return o;
}
json capabilities(const Options& o){
  long pages=sysconf(_SC_PHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
  uint64_t ram=pages>0&&page_size>0?uint64_t(pages)*uint64_t(page_size):512ull*1024*1024;
  uint64_t budget=o.memory?o.memory:ram/4,max_buffer=budget;
  std::string device="native CPU reference",backend="cpu";
  json limits=json::object();
  if(o.device!="cpu"){
    require(o.device=="vulkan"||o.device.rfind("vulkan:",0)==0,"device must be cpu or vulkan:N");
    unsigned index=o.device=="vulkan"?0:std::stoul(o.device.substr(7));
    auto devices=vulkan_devices();require(index<devices.size(),"Vulkan device index out of range");
    device=devices[index];backend="vulkan";auto info=vulkan_device_limits(index);limits=info;
    budget=std::min(budget,info.at("device_heap_bytes")/2);max_buffer=std::min(budget,info.at("max_buffer_bytes"));
  }
  return {{"abi",ABI},{"backend",backend},{"device",device},{"name",o.name},{"dtype","float32"},
          {"memory_bytes",budget},{"max_buffer_bytes",max_buffer},{"max_batch",o.max_batch},
          {"cpu_threads",std::max(1u,std::thread::hardware_concurrency())},{"physical_ram_bytes",ram},
          {"samples_per_second",o.rate},{"limits",limits}};
}
struct Executable {std::unique_ptr<Plan> plan;json program;std::string checkpoint;};
json cached_json(const fs::path& cache,const std::string& key){
  require(valid_hash(key),"invalid artifact hash");auto data=read_file(cache/key);
  require(sha(data)==key,"cached artifact checksum mismatch");auto value=json::parse(data);
  require(value.at("abi")==ABI,"artifact ABI mismatch");return value;
}
Executable build(const json& program,const json& checkpoint,const Options& o,const json& caps){
  Graph graph;std::map<Id,std::string> names;
  for(auto& p:program.at("parameters"))names.emplace(p.at("id").get<Id>(),p.at("name").get<std::string>());
  require(program.at("nodes").size()<=100000,"program too large");
  for(auto& n:program.at("nodes")){
    auto op=n.at("op").get<std::string>();Id id;
    if(op=="input"||op=="constant"||op=="parameter"){
      auto data=n.at("data").get<std::vector<float>>();
      if(op=="parameter"){
        auto p=checkpoint.at("parameters").at(names.at(Id(graph.nodes.size())));
        require(p.at("shape")==n.at("shape"),"parameter shape mismatch");data=p.at("data").get<std::vector<float>>();
      }
      for(auto f:data)require(std::isfinite(f),"non-finite program data");
      id=graph.leaf(op,n.at("name"),n.at("shape").get<Shape>(),data);
    } else id=graph.op(op,n.at("inputs").get<std::vector<Id>>(),n.at("attrs").get<Shape>());
    require(graph.at(id).shape==n.at("shape").get<Shape>(),"program shape inference mismatch");
  }
  auto plan=std::make_unique<Plan>(graph,program.at("outputs").get<std::vector<Id>>(),std::vector<std::pair<Id,Id>>{},true);
  auto arena=plan->stats().at("arena_bytes");
  require(arena==program.at("arena_bytes").get<uint64_t>(),"compiler arena ABI mismatch");
  require(arena<=caps.at("max_buffer_bytes").get<uint64_t>()&&arena*2<=caps.at("memory_bytes").get<uint64_t>(),"program exceeds node memory budget");
  require(plan->outputs.size()==program.at("parameters").size()+1&&numel(graph.at(plan->outputs[0]).shape)==1,"invalid gradient program outputs");
  require(plan->inputs.size()==program.at("feed_indices").size(),"program feed mapping mismatch");
  auto& hashes=program.at("shader_hashes");require(hashes.size()==plan->kernels.size(),"compiler kernel ABI mismatch");
  for(size_t i=0;i<plan->kernels.size();++i)require(sha(plan->kernels[i].source)==hashes.at(i),"compiler shader ABI mismatch; use matching owner/node versions");
  if(o.device!="cpu"){
    unsigned index=o.device=="vulkan"?0:std::stoul(o.device.substr(7));
    plan->enable_vulkan(program.at("spirv").get<std::vector<std::vector<uint32_t>>>(),index);
  }
  return {std::move(plan),program,{}};
}
json execute(const json& assignment,const fs::path& cache,const Options& o,const json& caps,
             std::map<std::string,Executable>& plans){
  auto program_key=assignment.at("program").get<std::string>();
  auto checkpoint_key=assignment.at("checkpoint").get<std::string>();
  auto checkpoint=cached_json(cache,checkpoint_key),data=cached_json(cache,assignment.at("data"));
  require(checkpoint.at("version")==assignment.at("base_version"),"checkpoint version mismatch");
  require(data.at("sample_ids").size()==assignment.at("samples").get<size_t>(),"sample manifest size mismatch");
  auto found=plans.find(program_key);
  if(found==plans.end()){
    // Bound resident execution plans: one profile at a time; disk cache is reusable.
    plans.clear();auto program=cached_json(cache,program_key);
    found=plans.emplace(program_key,build(program,checkpoint,o,caps)).first;
    found->second.checkpoint=checkpoint_key;
  }
  auto& e=found->second;
  if(e.checkpoint!=checkpoint_key){
    for(auto& p:e.program.at("parameters")){
      auto values=checkpoint.at("parameters").at(p.at("name").get<std::string>());
      require(values.at("shape")==p.at("shape"),"checkpoint parameter shape mismatch");
      auto array=values.at("data").get<std::vector<float>>();
      for(float v:array)require(std::isfinite(v),"non-finite checkpoint");
      e.plan->write(p.at("id").get<Id>(),array);
    }
    e.checkpoint=checkpoint_key;
  }
  std::vector<std::vector<float>> feeds;
  for(size_t i=0;i<e.plan->inputs.size();++i){
    auto input=data.at("inputs").at(e.program.at("feed_indices").at(i).get<size_t>());
    require(input.at("shape").get<Shape>()==e.plan->graph.at(e.plan->inputs[i]).shape,"data shape guard failed");
    auto values=input.at("data").get<std::vector<float>>();
    for(float v:values)require(std::isfinite(v),"non-finite input");feeds.push_back(std::move(values));
  }
  auto start=std::chrono::steady_clock::now();e.plan->run(feeds);
  float loss=e.plan->read(e.plan->outputs[0]).at(0);require(std::isfinite(loss),"non-finite loss");
  json gradients=json::object();
  for(size_t i=0;i<e.program.at("parameters").size();++i){
    auto& p=e.program.at("parameters").at(i);auto values=e.plan->read(e.plan->outputs.at(i+1));
    require(e.plan->graph.at(e.plan->outputs.at(i+1)).shape==p.at("shape").get<Shape>(),"gradient shape mismatch");
    for(float v:values)require(std::isfinite(v),"non-finite gradient");
    gradients[p.at("name").get<std::string>()]={{"shape",p.at("shape")},{"data",values}};
  }
  double seconds=std::max(1e-9,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  json result=assignment;result.erase("deadline");result["loss"]=loss;result["gradients"]=gradients;result["seconds"]=seconds;
  return result;
}
std::vector<fs::path> entries(const fs::path& directory){
  std::vector<fs::path> result;for(auto& e:fs::directory_iterator(directory))if(e.path().extension()==".json")result.push_back(e.path());
  std::sort(result.begin(),result.end());return result;
}
bool flush(Client& client,const fs::path& root){
  bool online=true;
  for(auto& path:entries(root/"outbox")){
    try {
      auto result=json::parse(read_file(path,MAX_RESPONSE));auto reply=client.post("/v1/results",result);
      auto status=reply.at("status").get<std::string>();
      require(status=="accepted"||status=="duplicate"||status=="already_completed","unrecognized result acknowledgement");
      std::cout<<status<<" "<<result.at("chunk_id").get<std::string>()<<std::endl;
      // Remove assignment first: if killed here, only the idempotent outbox remains.
      remove_durable(root/"assignments"/path.filename());remove_durable(path);
    }catch(const HttpError& e){
      if(e.status==400||e.status==409||e.status==422){
        fs::rename(path,root/"rejected"/path.filename());sync_directory(root/"rejected");sync_directory(root/"outbox");
        remove_durable(root/"assignments"/path.filename());std::cerr<<e.what()<<"; retained rejected result"<<std::endl;
      }else{online=false;std::cerr<<e.what()<<std::endl;break;}
    }catch(const std::exception& e){online=false;std::cerr<<e.what()<<std::endl;break;}
  }
  return online;
}
int run(const Options& o){
  auto owner=validate_owner(o.owner,o.allow_http);const char* raw=getenv("MUNET_SWARM_TOKEN");std::string token=raw?raw:"";
  require(token.size()>=32&&token.find_first_of("\r\n")==std::string::npos,"set MUNET_SWARM_TOKEN to at least 32 characters");
  fs::path root=o.state;for(auto name:{"cache","assignments","outbox","rejected","failed"})fs::create_directories(root/name);
  Lock lock(root/"node.lock");auto state_path=root/"worker.json";json state;
  if(fs::exists(state_path)){state=json::parse(read_file(state_path));require(state.at("owner")==owner,"state belongs to a different owner URL; use a new state directory");}
  else{state={{"owner",owner},{"worker_id",random_id()}};atomic_write(state_path,state.dump());}
  auto caps=capabilities(o);if(o.rate==0&&state.contains("samples_per_second"))caps["samples_per_second"]=state.at("samples_per_second");
  std::cout<<"MuNet node "<<state.at("worker_id").get<std::string>()<<" on "<<caps.at("device").get<std::string>()<<std::endl;
  Client client(owner,token);std::map<std::string,Executable> plans;bool registered=false;
  while(!stopping){
    bool compute_failed=false;
    if(!flush(client,root))registered=false;
    auto pending=entries(root/"assignments");
    if(pending.empty()&&entries(root/"outbox").empty()){
      try {
        if(!registered){
          auto reply=client.post("/v1/register",{{"worker_id",state.at("worker_id")},{"capabilities",caps}});
          if(state.contains("job_id"))require(state.at("job_id")==reply.at("job_id"),"owner serves a different job; use a new node state directory");
          state["job_id"]=reply.at("job_id");atomic_write(state_path,state.dump());registered=true;
        }
        auto reply=client.post("/v1/lease",{{"worker_id",state.at("worker_id")}});
        if(reply.at("state")=="completed"){std::cout<<"job completed"<<std::endl;return 0;}
        require(reply.at("assignments").size()<=64,"owner returned too many assignments");
        for(auto& a:reply.at("assignments")){
          require(a.at("abi")==ABI&&a.at("job_id")==state.at("job_id")&&a.at("worker_id")==state.at("worker_id"),"assignment identity mismatch");
          auto attempt=a.at("attempt").get<std::string>();
          require(attempt.size()==48&&attempt.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid attempt token");
          atomic_write(root/"assignments"/(attempt+".json"),a.dump());
        }
        pending=entries(root/"assignments");
      }catch(const std::exception& e){registered=false;std::cerr<<e.what()<<std::endl;}
    }
    // Prefetch independently. Completed downloads remain available through an outage.
    std::vector<std::pair<fs::path,json>> ready;
    for(auto& path:pending){
      if(fs::exists(root/"outbox"/path.filename()))continue;
      try {
        auto a=json::parse(read_file(path));
        for(auto key:{"program","checkpoint","data"})client.download(a.at(key),root/"cache");
        ready.emplace_back(path,std::move(a));
      }catch(const std::exception& e){registered=false;std::cerr<<e.what()<<std::endl;}
    }
    if(!ready.empty()){
      try {json attempts=json::array();for(auto& item:ready)attempts.push_back(item.second.at("attempt"));
        client.post("/v1/heartbeat",{{"worker_id",state.at("worker_id")},{"attempts",attempts}});
      }catch(const std::exception&){registered=false;} // Leases are an optimization, never a correctness dependency.
    }
    for(auto& item:ready){
      if(stopping)break;
      try {
        auto result=execute(item.second,root/"cache",o,caps,plans);
        auto text=result.dump();require(text.size()<=MAX_RESPONSE,"gradient result exceeds protocol limit");
        atomic_write(root/"outbox"/item.first.filename(),text);remove_durable(item.first);
        state["samples_per_second"]=std::min(1e12,result.at("samples").get<double>()/result.at("seconds").get<double>());
        caps["samples_per_second"]=state.at("samples_per_second");atomic_write(state_path,state.dump());
        std::cout<<"computed "<<result.at("chunk_id").get<std::string>()<<std::endl;
      }catch(const std::exception& e){
        compute_failed=true;
        fs::rename(item.first,root/"failed"/item.first.filename());sync_directory(root/"failed");sync_directory(root/"assignments");
        std::cerr<<"compute failed: "<<e.what()<<"; owner can retry after lease expiry"<<std::endl;
      }
    }
    if(!flush(client,root))registered=false;
    if(o.once)return compute_failed?1:(registered&&entries(root/"outbox").empty()&&entries(root/"assignments").empty()?0:2);
    // Poll between bundles; active execution and durable state survive connection loss.
    for(int i=0;i<int(std::ceil(o.poll*10))&&!stopping;++i)std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
}
}
int main(int argc,char** argv){
  try {
    require(curl_global_init(CURL_GLOBAL_DEFAULT)==CURLE_OK,"curl global initialization failed");
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);std::signal(SIGPIPE,SIG_IGN);
    int result=run(options(argc,argv));curl_global_cleanup();return result;
  }catch(const std::exception& e){std::cerr<<"munet-node: "<<e.what()<<std::endl;return 1;}
}
