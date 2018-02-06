package main

import (
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/bschieche/gomnist/controller"
	"github.com/bschieche/gomnist/manager"
	"github.com/go-kit/kit/log"
	"github.com/go-kit/kit/log/level"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	// init logger
	logger := log.NewLogfmtLogger(os.Stderr)
	logger = log.With(logger,
		"ts", log.DefaultTimestampUTC,
		"caller", log.DefaultCaller,
	)

	level.Info(logger).Log("msg", "starting ...")

	// load config
	cfg, err := newConfig()
	if err != nil {
		level.Error(logger).Log("msg", "failed to parse config", "err", err)
		os.Exit(1)
	}

	// initialize a manager and a corresponding controller
	m := manager.New()
	ctrl := controller.New(logger, m)

	// setup routes
	mux := http.NewServeMux()
	mux.HandleFunc("/render", ctrl.ServeRender)
	mux.HandleFunc("/submit", ctrl.ServeSubmit)
	mux.HandleFunc("/", ctrl.ServeIndex)

	// handle signals
	go handleSigs()

	// start metrics and pprof server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "OK", http.StatusOK)
		})
		http.HandleFunc("/", http.NotFound)
		if err := http.ListenAndServe(cfg.ListenMgmt, nil); err != nil {
			level.Warn(logger).Log("msg", "Failed to listen on management port", "err", err)
		}
	}()

	s := &http.Server{
		Addr:    cfg.Listen,
		Handler: prometheus.InstrumentHandler("goml", mux),
	}
	if err := s.ListenAndServe(); err != nil {
		logger.Log("level", "error", "msg", "Failed to listen", "err", err)
		os.Exit(1)
	}
}

func handleSigs() {
	exitChan := make(chan os.Signal, 10)
	signal.Notify(exitChan, syscall.SIGINT, syscall.SIGTERM)

	<-exitChan
	os.Exit(0)
}
