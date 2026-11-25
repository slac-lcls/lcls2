#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sched.h>
#include <pthread.h>
#include <time.h>

using Clock = std::chrono::steady_clock;
using U = uint64_t;
static void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
static std::string quote(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
static U number(const char* s) {
    check(*s && *s != '-', "invalid unsigned number");
    char* end; errno = 0; U n = std::strtoull(s, &end, 10);
    check(!errno && !*end, "invalid unsigned number"); return n;
}
struct FD {
    int fd;
    explicit FD(const std::string& path): fd(open(path.c_str(), O_RDONLY | O_NOFOLLOW)) { check(fd >= 0, "open: " + path + ": " + strerror(errno)); }
    ~FD() { close(fd); }
    FD(const FD&) = delete;
};
struct Id {
    U dev, ino, size, mtime;
    bool operator==(const Id& x) const { return dev==x.dev && ino==x.ino && size==x.size && mtime==x.mtime; }
};
static Id identify(int fd) {
    struct stat s; check(fstat(fd, &s)==0, "fstat failed");
    check(S_ISREG(s.st_mode) && s.st_nlink==1 && s.st_size>0, "input must be nonempty private regular file");
    return {U(s.st_dev),U(s.st_ino),U(s.st_size),U(s.st_mtim.tv_sec)*1000000000ULL+U(s.st_mtim.tv_nsec)};
}
static std::string id_json(const Id& x) {
    std::ostringstream o; o << "{\"st_dev\":"<<x.dev<<",\"st_ino\":"<<x.ino<<",\"st_size\":"<<x.size<<",\"st_mtime_ns\":"<<x.mtime<<"}"; return o.str();
}
static double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b-a).count(); }
static double cpu_seconds() {
    timespec t; check(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)==0, "thread clock failed");
    return double(t.tv_sec)+double(t.tv_nsec)*1e-9;
}
static U resident(int fd, U size, U page) {
    constexpr U window=256ULL*1024*1024; U count=0;
    for (U off=0; off<size; off+=window) {
        size_t length=std::min(window,size-off); std::vector<unsigned char> bits((length+page-1)/page);
        void* p=mmap(nullptr,length,PROT_READ,MAP_SHARED,fd,off);
        check(p!=MAP_FAILED,"mmap for mincore failed");
        int rc=mincore(p,length,bits.data()); int saved=errno;
        int unmap=munmap(p,length);
        check(rc==0,"mincore failed: "+std::string(strerror(saved))); check(unmap==0,"munmap failed");
        for (auto b:bits) count+=(b&1);
    }
    return count;
}
struct Result {
    size_t file; int cpu, actual_start=-1, actual_end=-1;
    U begin,end,bytes=0,calls=0,checksum=0;
    double cpu_time=0;
    Clock::time_point start,finish;
    std::string error;
};
int main(int argc,char** argv) {
    try {
        if(argc>=3 && std::string(argv[1])=="--evict-all") {
            U page=sysconf(_SC_PAGESIZE); check(page>0 && page<=65536,"invalid page size");
            std::vector<std::unique_ptr<FD>> files; std::vector<Id> ids; std::vector<U> counts;
            for(int i=2;i<argc;++i) { files.emplace_back(new FD(argv[i])); ids.push_back(identify(files.back()->fd)); counts.push_back(0); }
            U attempt;
            for(attempt=1;attempt<=3;++attempt) {
                for(size_t i=0;i<files.size();++i) check(posix_fadvise(files[i]->fd,0,((ids[i].size+page-1)/page)*page,POSIX_FADV_DONTNEED)==0,"global fadvise failed");
                U total=0;
                for(size_t i=0;i<files.size();++i) { counts[i]=resident(files[i]->fd,ids[i].size,page); total+=counts[i]; }
                if(total) continue;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                for(size_t i=0;i<files.size();++i) { counts[i]=resident(files[i]->fd,ids[i].size,page); total+=counts[i]; }
                if(!total) break;
            }
            check(attempt<=3,"global cache eviction failed");
            std::cout<<"{\"state\":\"complete\",\"confirmed_zero_scans\":2,\"files\":[";
            for(size_t i=0;i<files.size();++i) {
                check(identify(files[i]->fd)==ids[i],"input changed during global eviction");
                if(i)std::cout<<',';
                std::cout<<"{\"path\":"<<quote(argv[i+2])<<",\"identity\":"<<id_json(ids[i])<<",\"resident_pages\":"<<counts[i]<<"}";
            }
            std::cout<<"]}\n"; return 0;
        }
        check(argc>=6,"usage: reader THREADS REQUEST_BYTES CPU_CSV WARM_PROBE(0/1) FILE...");
        U n=number(argv[1]),request=number(argv[2]),warm=number(argv[4]); size_t nf=argc-5;
        check(n>0 && n<=256 && nf<=64 && n%nf==0,"invalid worker/file count");
        check(request>=4096 && request<=64*1024*1024 && warm<=1,"invalid request size or warm probe");
        U page=sysconf(_SC_PAGESIZE); check(page>0 && page<=65536,"invalid page size");
        std::vector<int> cpus; std::stringstream csv(argv[3]); std::string token;
        while (std::getline(csv,token,',')) { U c=number(token.c_str()); check(c<CPU_SETSIZE,"CPU exceeds CPU_SETSIZE"); cpus.push_back(int(c)); }
        check(cpus.size()==n && std::set<int>(cpus.begin(),cpus.end()).size()==n,"CPU list must be unique and match workers");
        cpu_set_t allowed; check(sched_getaffinity(0,sizeof(allowed),&allowed)==0,"getaffinity failed");
        for(int c:cpus) check(CPU_ISSET(c,&allowed),"CPU not in allocation affinity mask");
        std::vector<std::string> paths; std::vector<std::unique_ptr<FD>> files; std::vector<Id> ids;
        std::set<std::pair<U,U>> inodes;
        for(int i=5;i<argc;++i) {
            paths.emplace_back(argv[i]); files.emplace_back(new FD(paths.back())); ids.push_back(identify(files.back()->fd));
            check(inodes.emplace(ids.back().dev,ids.back().ino).second,"duplicate input inode");
        }
        std::vector<Result> results(n); U per_file=n/nf;
        for(U i=0;i<n;++i) {
            auto& r=results[i]; r.file=i/per_file; r.cpu=cpus[i]; U part=i%per_file;
            U chunks=(ids[r.file].size+request-1)/request;
            r.begin=std::min(ids[r.file].size,(chunks*part/per_file)*request);
            r.end=std::min(ids[r.file].size,(chunks*(part+1)/per_file)*request);
        }
        std::mutex mutex; std::condition_variable cv; U ready=0; bool go=false,abort=false;
        std::vector<std::thread> threads; Clock::time_point release;
        std::vector<U> warmed(nf,0),cold(nf,0); U attempts=0; double eviction_seconds=0;
        auto worker=[&](U index) {
            auto& r=results[index]; bool registered=false;
            try {
                cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(r.cpu,&mask);
                check(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)==0,"worker setaffinity failed");
                FD fd(paths[r.file]); check(identify(fd.fd)==ids[r.file],"input replaced before worker open");
                void* raw=nullptr; check(posix_memalign(&raw,4096,request)==0,"buffer allocation failed");
                std::unique_ptr<void,decltype(&free)> buffer(raw,free); memset(raw,0,request);
#ifdef READER_TESTING
                std::string fault=getenv("READER_TEST_FAULT")?getenv("READER_TEST_FAULT"):"";
                if(index==0 && fault=="setup") throw std::runtime_error("injected setup failure");
#endif
                { std::unique_lock<std::mutex> lock(mutex); ++ready; registered=true; cv.notify_all();
                  cv.wait(lock,[&]{return go;}); if(abort) return; }
                r.start=Clock::now(); r.actual_start=sched_getcpu(); double cpu_begin=cpu_seconds();
                for(U pos=r.begin;pos<r.end;) {
                    U wanted=std::min(request,r.end-pos),done=0;
                    while(done<wanted) {
                        ssize_t got;
#ifdef READER_TESTING
                        ++r.calls;
                        if(index==0 && fault=="eintr" && r.calls==1) { got=-1; errno=EINTR; }
                        else if(index==0 && fault=="eof") got=0;
                        else if(index==0 && fault=="readerror") { got=-1; errno=EIO; }
                        else got=pread(fd.fd,static_cast<char*>(raw)+done,
                              fault=="short"?std::min<U>(17,wanted-done):wanted-done,pos+done);
#else
                        ++r.calls; got=pread(fd.fd,static_cast<char*>(raw)+done,wanted-done,pos+done);
#endif
                        if(got<0 && errno==EINTR) continue;
                        check(got>0,got==0?"unexpected EOF":"pread failed: "+std::string(strerror(errno)));
#ifdef READER_TESTING
                        for(ssize_t j=0;j<got;++j) r.checksum+=static_cast<unsigned char*>(raw)[done+j];
#endif
                        done+=U(got); r.bytes+=U(got);
                    }
                    pos+=done;
                }
                r.finish=Clock::now(); r.cpu_time=cpu_seconds()-cpu_begin; r.actual_end=sched_getcpu();
                check(r.actual_start==r.cpu && r.actual_end==r.cpu,"worker affinity mismatch");
                check(r.bytes==r.end-r.begin,"worker byte mismatch");
                check(identify(fd.fd)==ids[r.file],"input changed during read");
            } catch(const std::exception& e) {
                r.error=e.what(); std::lock_guard<std::mutex> lock(mutex);
                if(!registered) ++ready;
                cv.notify_all();
            }
        };
        try {
            for(U i=0;i<n;++i) threads.emplace_back(worker,i);
            { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock,[&]{return ready==n;}); }
            for(const auto& r:results) check(r.error.empty(),r.error);
            auto evict_start=Clock::now();
            if(warm) {
                for(size_t f=0;f<nf;++f) {
                    check(posix_fadvise(files[f]->fd,0,0,POSIX_FADV_RANDOM)==0,"probe readahead control failed");
                    char b; check(pread(files[f]->fd,&b,1,0)==1,"warm probe failed");
                    warmed[f]=resident(files[f]->fd,ids[f].size,page);
                    check(warmed[f]>0,"mincore positive control failed");
                }
                bool stable=false;
                for(int tries=0;tries<3 && !stable;++tries) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    stable=true;
                    for(size_t f=0;f<nf;++f) { U now=resident(files[f]->fd,ids[f].size,page); stable=stable && now==warmed[f]; warmed[f]=now; }
                }
                check(stable,"warm probe residency did not stabilize");
            }
            for(attempts=1;attempts<=3;++attempts) {
                for(size_t f=0;f<nf;++f) {
                    U rounded=((ids[f].size+page-1)/page)*page;
                    int rc=posix_fadvise(files[f]->fd,0,rounded,POSIX_FADV_DONTNEED);
                    check(rc==0,"fadvise DONTNEED failed: "+std::string(strerror(rc)));
                }
                U total=0;
                for(size_t f=0;f<nf;++f) { cold[f]=resident(files[f]->fd,ids[f].size,page); total+=cold[f]; }
#ifdef READER_TESTING
                if(getenv("READER_TEST_FAULT") && std::string(getenv("READER_TEST_FAULT"))=="cache") total=1;
#endif
                if(!total) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    for(size_t f=0;f<nf;++f) { cold[f]=resident(files[f]->fd,ids[f].size,page); total+=cold[f]; }
                    if(!total) break;
                }
            }
            check(attempts<=3,"local page cache not empty after three eviction attempts");
            eviction_seconds=seconds(evict_start,Clock::now());
            { std::lock_guard<std::mutex> lock(mutex); release=Clock::now(); go=true; }
            cv.notify_all();
        } catch(...) {
            { std::lock_guard<std::mutex> lock(mutex); abort=true; go=true; }
            cv.notify_all(); for(auto& t:threads) if(t.joinable()) t.join(); throw;
        }
        for(auto& t:threads) t.join();
        U total=0,expected=0; auto finish=release;
        for(size_t f=0;f<nf;++f) { check(identify(files[f]->fd)==ids[f],"input identity changed"); expected+=ids[f].size; }
        for(const auto& r:results) { check(r.error.empty(),r.error); total+=r.bytes; finish=std::max(finish,r.finish); }
        check(total==expected,"aggregate byte mismatch"); double elapsed=seconds(release,finish);
        check(elapsed>0,"nonpositive elapsed");
        std::cout << std::setprecision(17) << "{\"state\":\"complete\",\"threads\":"<<n<<",\"request_bytes\":"<<request
                  <<",\"bytes\":"<<total<<",\"seconds\":"<<elapsed<<",\"gib_per_second\":"<<double(total)/elapsed/1073741824.0
                  <<",\"page_size\":"<<page<<",\"eviction_attempts\":"<<attempts<<",\"eviction_seconds\":"<<eviction_seconds
                  <<",\"confirmed_zero_scans\":2"
                  <<",\"warm_probe\":"<<(warm?"true":"false")<<",\"files\":[";
        for(size_t f=0;f<nf;++f) {
            if(f) std::cout<<',';
            std::cout<<"{\"path\":"<<quote(paths[f])<<",\"identity\":"<<id_json(ids[f])<<",\"pages\":"<<(ids[f].size+page-1)/page
                     <<",\"warm_resident_pages\":"<<warmed[f]<<",\"cold_resident_pages\":"<<cold[f]<<"}";
        }
        std::cout<<"],\"workers\":[";
        for(U i=0;i<n;++i) {
            const auto& r=results[i]; if(i)std::cout<<',';
            std::cout<<"{\"index\":"<<i<<",\"file\":"<<r.file<<",\"cpu\":"<<r.cpu<<",\"actual_start_cpu\":"<<r.actual_start
                     <<",\"actual_end_cpu\":"<<r.actual_end<<",\"begin\":"<<r.begin<<",\"end\":"<<r.end<<",\"bytes\":"<<r.bytes
                     <<",\"calls\":"<<r.calls<<",\"start_seconds\":"<<seconds(release,r.start)<<",\"end_seconds\":"<<seconds(release,r.finish)
                     <<",\"cpu_seconds\":"<<r.cpu_time<<",\"test_checksum\":"<<r.checksum<<"}";
        }
        std::cout<<"]}\n"; return 0;
    } catch(const std::exception& e) {
        std::cout<<"{\"state\":\"failed\",\"error\":"<<quote(e.what())<<"}\n"; return 1;
    }
}
