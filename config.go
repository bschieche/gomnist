package main

import "github.com/alexflint/go-arg"

type config struct {
	Listen     string `arg:"--listen,env:LISTEN"`
	ListenMgmt string `arg:"--listen-mgmt,env:LISTEN_MGMT"`
}

func newConfig() (*config, error) {
	c := &config{
		Listen:     ":8080",
		ListenMgmt: ":8081",
	}
	err := arg.Parse(c)
	return c, err
}
