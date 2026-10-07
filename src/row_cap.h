#ifndef MYNAH_ROW_CAP_H
#define MYNAH_ROW_CAP_H

/* The one storage bound for rows stepped together: the driver's job and
 * active-slot ceilings (graph.h), the Pocket engine's per-batch arrays
 * (engine_pocket.c) and the CUDA batch-metadata and graph caches
 * (backend_cuda.cu) all derive from it.  384 is the measured default.  A
 * CUDA build may raise it at compile time (`make cuda cuda-server ROW_CAP=768`)
 * for a GPU whose memory and compute hold more streams; the stack arrays sized
 * by it grow with it, and the server's --max-batch / --max-inflight clamp to
 * it.  It is a build-time constant on purpose: the arrays it sizes are on the
 * stack of the scheduler thread. */
#ifndef MYNAH_ROW_CAP
#define MYNAH_ROW_CAP 384u
#endif

#endif
