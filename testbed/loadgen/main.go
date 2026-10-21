// loadgen: open-loop replay of per-minute request counts against several targets at once.
//
// Every target replays the schedule of its function: in minute m it sends exactly count[m]
// requests at times drawn uniformly at random within the minute (the same times for every
// target of a function). Responses do not slow the sender down. Keep-alive is disabled, so
// every request opens a new connection and the Service spreads requests over ready pods.
//
// Outputs:
//   load.csv  one row per target and second of planned send time: planned, dispatched,
//             outcomes, dispatch lag, and a latency histogram of successful (HTTP 200) requests
//   served.csv  per target and pod: first and last successful response, number served
// GET /stats returns, per target, the requests dispatched in every completed minute; the
// controllers read it as the function's measured request count.
package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type target struct {
	Name     string `json:"name"`
	Function string `json:"function"`
	URL      string `json:"url"`
}

type config struct {
	Schedules map[string]string `json:"schedules"`
	Targets   []target          `json:"targets"`
	Seed      int64             `json:"seed"`
	Minutes   int               `json:"minutes"`
	Interval  float64           `json:"interval_seconds"`
}

var bounds = []float64{5, 10, 25, 50, 100, 250, 500, 1000, 2000}

type second struct {
	planned, dispatched, ok, shed, timeout, failed atomic.Int64
	lagSumUs, lagMaxUs                            atomic.Int64
	hist                                          [10]atomic.Int64
}

type served struct {
	first, last float64
	n           int64
}

func readSchedule(path string) []int {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	out := make([]int, 0, len(rows))
	for _, r := range rows[1:] {
		v, err := strconv.Atoi(r[0])
		if err != nil {
			log.Fatal(err)
		}
		out = append(out, v)
	}
	return out
}

func offsets(counts []int, interval float64, seed int64) [][]time.Duration {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]time.Duration, len(counts))
	for m, n := range counts {
		ts := make([]float64, n)
		for i := range ts {
			ts[i] = rng.Float64() * interval
		}
		sort.Float64s(ts)
		out[m] = make([]time.Duration, n)
		for i, t := range ts {
			out[m][i] = time.Duration(t * float64(time.Second))
		}
	}
	return out
}

func maxStore(a *atomic.Int64, v int64) {
	for {
		old := a.Load()
		if v <= old || a.CompareAndSwap(old, v) {
			return
		}
	}
}

func main() {
	cfgPath := flag.String("config", "loadgen.json", "configuration file")
	start := flag.Int64("start", 0, "unix time of minute 0 (0: next full minute)")
	out := flag.String("out", "load.csv", "per-second output CSV")
	servedOut := flag.String("served", "served.csv", "per-pod output CSV")
	listen := flag.String("listen", ":9100", "stats address")
	timeout := flag.Duration("timeout", 3*time.Second, "request timeout")
	flag.Parse()

	raw, err := os.ReadFile(*cfgPath)
	if err != nil {
		log.Fatal(err)
	}
	var cfg config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		log.Fatal(err)
	}
	interval := time.Duration(cfg.Interval * float64(time.Second))
	plans := map[string][][]time.Duration{}
	fnames := make([]string, 0, len(cfg.Schedules))
	for fn := range cfg.Schedules {
		fnames = append(fnames, fn)
	}
	sort.Strings(fnames)
	for i, fn := range fnames {
		plans[fn] = offsets(readSchedule(cfg.Schedules[fn])[:cfg.Minutes], cfg.Interval, cfg.Seed+int64(i))
	}

	t0 := time.Unix(*start, 0)
	if *start == 0 {
		t0 = time.Now().Truncate(time.Minute).Add(time.Minute)
	}
	nsec := int(float64(cfg.Minutes)*cfg.Interval) + 1
	stats := make([][]second, len(cfg.Targets))
	sent := make([][]atomic.Int64, len(cfg.Targets))
	done := make([]atomic.Int64, len(cfg.Targets))
	pods := make([]map[string]*served, len(cfg.Targets))
	var podMu sync.Mutex
	for i, tg := range cfg.Targets {
		stats[i] = make([]second, nsec)
		sent[i] = make([]atomic.Int64, cfg.Minutes)
		pods[i] = map[string]*served{}
		for m, offs := range plans[tg.Function] {
			for _, off := range offs {
				sec := int((time.Duration(m)*interval + off) / time.Second)
				stats[i][sec].planned.Add(1)
			}
		}
	}

	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		res := map[string]any{"t0": t0.Unix(), "interval_seconds": cfg.Interval, "minutes": cfg.Minutes}
		per := map[string][]int64{}
		for i, tg := range cfg.Targets {
			k := int(done[i].Load())
			v := make([]int64, k)
			for j := 0; j < k; j++ {
				v[j] = sent[i][j].Load()
			}
			per[tg.Name] = v
		}
		res["arrivals"] = per
		json.NewEncoder(w).Encode(res)
	})
	go func() { log.Fatal(http.ListenAndServe(*listen, nil)) }()

	client := &http.Client{Timeout: *timeout, Transport: &http.Transport{DisableKeepAlives: true}}
	var senders, inflight sync.WaitGroup
	log.Printf("minute 0 at %s, %d targets, %d minutes", t0.Format(time.RFC3339), len(cfg.Targets), cfg.Minutes)
	for i, tg := range cfg.Targets {
		senders.Add(1)
		go func(i int, tg target) {
			defer senders.Done()
			plan := plans[tg.Function]
			for m := 0; m < cfg.Minutes; m++ {
				base := t0.Add(time.Duration(m) * interval)
				for _, off := range plan[m] {
					at := base.Add(off)
					time.Sleep(time.Until(at))
					now := time.Now()
					lag := now.Sub(at).Microseconds()
					if lag < 0 {
						lag = 0
					}
					s := &stats[i][int(at.Sub(t0)/time.Second)]
					s.dispatched.Add(1)
					s.lagSumUs.Add(lag)
					maxStore(&s.lagMaxUs, lag)
					sent[i][m].Add(1)
					inflight.Add(1)
					go func(sentAt time.Time, s *second) {
						defer inflight.Done()
						resp, err := client.Get(tg.URL)
						if err != nil {
							var ne net.Error
							if errors.As(err, &ne) && ne.Timeout() {
								s.timeout.Add(1)
							} else {
								s.failed.Add(1)
							}
							return
						}
						io.Copy(io.Discard, resp.Body)
						resp.Body.Close()
						lat := float64(time.Since(sentAt).Microseconds()) / 1000
						switch resp.StatusCode {
						case http.StatusOK:
							s.ok.Add(1)
							s.hist[sort.SearchFloat64s(bounds, lat)].Add(1)
							if pod := resp.Header.Get("X-Pod"); pod != "" {
								t := float64(time.Now().UnixNano()) / 1e9
								podMu.Lock()
								p := pods[i][pod]
								if p == nil {
									p = &served{first: t}
									pods[i][pod] = p
								}
								p.last = t
								p.n++
								podMu.Unlock()
							}
						case http.StatusServiceUnavailable:
							s.shed.Add(1)
						default:
							s.failed.Add(1)
						}
					}(now, s)
				}
				done[i].Store(int64(m + 1))
			}
		}(i, tg)
	}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	w := bufio.NewWriter(f)
	fmt.Fprint(w, "second,target,planned,dispatched,ok,shed,timeout,failed,lag_ms_mean,lag_ms_max")
	for _, b := range bounds {
		fmt.Fprintf(w, ",ok_le%g", b)
	}
	fmt.Fprintln(w, ",ok_gt2000")
	written := 0
	flush := func(upto int) {
		for ; written < upto && written < nsec; written++ {
			for i, tg := range cfg.Targets {
				s := &stats[i][written]
				d := s.dispatched.Load()
				mean := 0.0
				if d > 0 {
					mean = float64(s.lagSumUs.Load()) / float64(d) / 1000
				}
				fmt.Fprintf(w, "%d,%s,%d,%d,%d,%d,%d,%d,%.3f,%.3f", written, tg.Name, s.planned.Load(), d, s.ok.Load(),
					s.shed.Load(), s.timeout.Load(), s.failed.Load(), mean, float64(s.lagMaxUs.Load())/1000)
				for b := range s.hist {
					fmt.Fprintf(w, ",%d", s.hist[b].Load())
				}
				fmt.Fprintln(w)
			}
		}
		w.Flush()
	}
	writeServed := func() {
		sf, err := os.Create(*servedOut)
		if err != nil {
			log.Print(err)
			return
		}
		sw := bufio.NewWriter(sf)
		fmt.Fprintln(sw, "target,pod,first_ok,last_ok,served")
		podMu.Lock()
		for i, tg := range cfg.Targets {
			for pod, p := range pods[i] {
				fmt.Fprintf(sw, "%s,%s,%.3f,%.3f,%d\n", tg.Name, pod, p.first, p.last, p.n)
			}
		}
		podMu.Unlock()
		sw.Flush()
		sf.Close()
	}
	finished := make(chan struct{})
	go func() { senders.Wait(); inflight.Wait(); close(finished) }()
	tick := time.NewTicker(time.Second)
	ticks := 0
	for {
		select {
		case <-tick.C:
			flush(int(time.Since(t0)/time.Second) - int(timeout.Seconds()) - 5)
			ticks++
			if ticks%60 == 0 {
				writeServed()
			}
		case <-finished:
			flush(nsec)
			f.Close()
			writeServed()
			log.Printf("done")
			select {}
		}
	}
}
