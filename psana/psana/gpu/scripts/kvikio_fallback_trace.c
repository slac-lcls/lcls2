/* Benchmark-only KvikIO 24.08 fallback tracing. No additional CUDA waits.
 * Build: cc -O2 -std=gnu11 -shared -fPIC -o fallback.so this.c -ldl -pthread
 * LD_PRELOAD catches POSIX pread64. install() replaces ONLY the two verified
 * function pointers in KvikIO's CUDA shim; installed files stay untouched.
 * Records are buffered in RAM and written after the timed loop has drained.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t start, end, size;
    int64_t offset, result;
    uint32_t tid, kind, fd, batch, read_id, reserved;
} Record;
_Static_assert(sizeof(Record) == 64, "trace layout");
static Record *records;
static const unsigned capacity = 2000000;
static _Atomic unsigned count, active, batch, overflow;
static __thread Record context;
static __thread int in_read;
static __thread unsigned tid;
static ssize_t (*real_pread)(int, void *, size_t, off64_t);
static int (*real_copy)(uint64_t, const void *, size_t, void *);
static int (*real_sync)(void *);
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void resolve(void) { real_pread = dlsym(RTLD_NEXT, "pread64"); if (!real_pread) abort(); }
static uint64_t now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
static unsigned put(Record r) {
    unsigned i = atomic_fetch_add_explicit(&count, 1, memory_order_relaxed);
    if (i < capacity) records[i] = r;
    else atomic_store(&overflow, 1);
    return i;
}
ssize_t pread64(int fd, void *buf, size_t n, off64_t off) {
    pthread_once(&once, resolve);
    Dl_info info;
    int capture = atomic_load_explicit(&active, memory_order_relaxed) &&
        dladdr(__builtin_return_address(0), &info) && info.dli_fname &&
        strstr(info.dli_fname, "/kvikio/_lib/libkvikio.");
    if (!capture) return real_pread(fd, buf, n, off);
    if (!tid) tid = (unsigned)syscall(SYS_gettid);
    Record r = {.size=n, .offset=off, .tid=tid, .kind=1, .fd=fd,
                .batch=atomic_load(&batch), .read_id=0, .reserved=0};
    r.start = now();
    ssize_t ret = real_pread(fd, buf, n, off);
    int saved_errno = errno;
    r.end = now(); r.result = ret;
    r.read_id = put(r);
    /* The read record's own ID is its array index; copies carry that index. */
    context = r; in_read = ret > 0;
    errno = saved_errno;
    return ret;
}
static int copy_hook(uint64_t dst, const void *src, size_t n, void *stream) {
    if (!in_read || !atomic_load(&active)) return real_copy(dst, src, n, stream);
    Record r = context; r.kind = 2; r.size = n; r.start = now();
    int ret = real_copy(dst, src, n, stream);
    r.end = now(); r.result = ret; put(r);
    return ret;
}
static int sync_hook(void *stream) {
    if (!in_read || !atomic_load(&active)) return real_sync(stream);
    Record r = context; r.kind = 3; r.start = now();
    int ret = real_sync(stream);
    r.end = now(); r.result = ret; put(r); in_read = 0;
    return ret;
}
int fallback_install(const char *path) {
    if (records) return -1;
    void *lib = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib || !cuda) return -2;
    void **(*instance)(void) = dlsym(lib, "_ZN6kvikio7cudaAPI8instanceEv");
    if (!instance) return -3;
    void **api = instance();
    /* Exact field order verified against installed shim/cuda.hpp. Fail closed. */
    if (api[3] != dlsym(cuda, "cuMemcpyHtoDAsync_v2") ||
        api[16] != dlsym(cuda, "cuStreamSynchronize")) return -4;
    records = calloc(capacity, sizeof(Record));
    if (!records) return -5;
    /* Fault pages in before the timed loop. */
    for (size_t i=0; i<capacity*sizeof(Record); i+=4096) ((volatile char *)records)[i]=0;
    real_copy = api[3]; real_sync = api[16];
    api[3] = copy_hook; api[16] = sync_hook;
    return 0;
}
void fallback_begin(void) { atomic_store(&count, 0); atomic_store(&overflow, 0); atomic_store(&batch, 0); atomic_store(&active, 1); }
void fallback_batch(unsigned value) { atomic_store(&batch, value); }
void fallback_end(void) { atomic_store(&active, 0); }
int fallback_dump(const char *path) {
    if (atomic_load(&active) || atomic_load(&overflow)) return -1;
    FILE *f = fopen(path, "wbx");
    if (!f) return -2;
    unsigned n = atomic_load(&count);
    int ok = fwrite(records, sizeof(Record), n, f) == n;
    int rc = fclose(f);
    return ok && !rc ? (int)n : -3;
}
