// Exact batched pread gather for PLE rows. No caching, no reordering of bytes.
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

typedef struct {
  const int32_t *fds;
  const int64_t *offs;
  const int64_t *lens;
  const int64_t *outs;
  char *out;
  int64_t begin;
  int64_t end;
  int err;
} ple_arg_t;

static void *ple_worker(void *p) {
  ple_arg_t *a = (ple_arg_t *)p;
  for (int64_t i = a->begin; i < a->end; i++) {
    int64_t done = 0;
    while (done < a->lens[i]) {
      ssize_t r = pread(a->fds[i], a->out + a->outs[i] + done,
                        (size_t)(a->lens[i] - done), (off_t)(a->offs[i] + done));
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) {
        a->err = r < 0 ? errno : -1;
        return NULL;
      }
      done += r;
    }
  }
  return NULL;
}

// Returns 0 on success, errno (>0) on read error, -1 on short read, -2 on bad args.
int ple_gather(const int32_t *fds, const int64_t *offs, const int64_t *lens,
               const int64_t *outs, char *out, int64_t n, int nthreads) {
  if (n < 0 || nthreads < 1 || nthreads > 256) return -2;
  if (n == 0) return 0;
  if (nthreads > n) nthreads = (int)n;
  pthread_t th[256];
  int created[256];
  ple_arg_t args[256];
  for (int t = 0; t < nthreads; t++) {
    args[t].fds = fds; args[t].offs = offs; args[t].lens = lens; args[t].outs = outs;
    args[t].out = out; args[t].err = 0;
    args[t].begin = n * t / nthreads;
    args[t].end = n * (t + 1) / nthreads;
    created[t] = 0;
  }
  for (int t = 1; t < nthreads; t++)
    created[t] = pthread_create(&th[t], NULL, ple_worker, &args[t]) == 0;
  ple_worker(&args[0]);
  for (int t = 1; t < nthreads; t++) {
    if (created[t]) pthread_join(th[t], NULL);
    else ple_worker(&args[t]);  // creation failed: same slice, run inline
  }
  for (int t = 0; t < nthreads; t++)
    if (args[t].err) return args[t].err;
  return 0;
}
