// worker: a CPU-bound HTTP service that stands in for one serverless function.
//
//   GET /work     burns a fixed amount of CPU (ITER sha256 rounds); more than MAX_INFLIGHT
//                 concurrent requests are rejected with 503 (load shedding)
//   GET /healthz  readiness probe: 503 until WARMUP_SECONDS have passed since start
//
// The per-pod capacity is measured on the testbed (testbed/capacity.json).
package main

import (
	"crypto/sha256"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync/atomic"
	"time"
)

var (
	start  = time.Now()
	iters  = envInt("ITER", 20000)
	warmup = time.Duration(envInt("WARMUP_SECONDS", 0)) * time.Second
	served atomic.Int64
	shed   atomic.Int64
	slots = make(chan struct{}, envInt("MAX_INFLIGHT", 8))
	host, _ = os.Hostname()
)

func envInt(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}

func work(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	select {
	case slots <- struct{}{}:
		defer func() { <-slots }()
	default:
		shed.Add(1)
		w.Header().Set("X-Pod", host)
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
		return
	}
	b := []byte(host)
	var s [32]byte
	for i := 0; i < iters; i++ {
		s = sha256.Sum256(b)
		b = s[:]
	}
	n := served.Add(1)
	w.Header().Set("X-Pod", host)
	w.Header().Set("X-Service-Ms", fmt.Sprintf("%.2f", float64(time.Since(t0).Microseconds())/1000))
	fmt.Fprintf(w, "%s %d %x\n", host, n, s[:4])
}

func healthz(w http.ResponseWriter, r *http.Request) {
	if time.Since(start) < warmup {
		http.Error(w, "warming up", http.StatusServiceUnavailable)
		return
	}
	w.Write([]byte("ok"))
}

func main() {
	log.Printf("pod %s starting, warmup=%s iters=%d", host, warmup, iters)
	http.HandleFunc("/work", work)
	http.HandleFunc("/healthz", healthz)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
