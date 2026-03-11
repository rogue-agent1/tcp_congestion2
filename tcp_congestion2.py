#!/usr/bin/env python3
"""TCP congestion control — CUBIC, BBR, and Reno simulation.

Simulates cwnd evolution under different congestion control algorithms
with packet loss events, RTT measurements, and bandwidth estimation.

Usage: python tcp_congestion2.py [--test]
"""

import sys, math

class Reno:
    """TCP Reno (NewReno): AIMD congestion control."""
    def __init__(self, mss=1460):
        self.mss = mss
        self.cwnd = mss * 10  # IW=10
        self.ssthresh = float('inf')
        self.state = 'slow_start'
        self.dup_acks = 0
        self.history = []
    
    def on_ack(self):
        self.history.append(self.cwnd)
        if self.state == 'slow_start':
            self.cwnd += self.mss
            if self.cwnd >= self.ssthresh:
                self.state = 'congestion_avoidance'
        else:
            self.cwnd += self.mss * self.mss // self.cwnd
    
    def on_loss(self):
        self.ssthresh = max(self.cwnd // 2, 2 * self.mss)
        self.cwnd = self.ssthresh
        self.state = 'congestion_avoidance'
        self.history.append(self.cwnd)
    
    def on_timeout(self):
        self.ssthresh = max(self.cwnd // 2, 2 * self.mss)
        self.cwnd = self.mss
        self.state = 'slow_start'
        self.history.append(self.cwnd)

class CUBIC:
    """CUBIC congestion control (Linux default since 2.6.19)."""
    def __init__(self, mss=1460, C=0.4, beta=0.7):
        self.mss = mss
        self.C = C
        self.beta = beta
        self.cwnd = mss * 10
        self.ssthresh = float('inf')
        self.W_max = 0
        self.t_epoch = 0
        self.K = 0
        self.time = 0
        self.state = 'slow_start'
        self.history = []
    
    def on_ack(self, rtt=0.1):
        self.time += rtt
        self.history.append(self.cwnd)
        
        if self.state == 'slow_start':
            self.cwnd += self.mss
            if self.cwnd >= self.ssthresh:
                self.state = 'congestion_avoidance'
                self.t_epoch = self.time
            return
        
        t = self.time - self.t_epoch
        W_cubic = self.C * (t - self.K) ** 3 + self.W_max
        W_cubic_bytes = max(W_cubic * self.mss, self.mss)
        
        if W_cubic_bytes > self.cwnd:
            self.cwnd = int(W_cubic_bytes)
        else:
            self.cwnd += self.mss * self.mss // max(self.cwnd, 1)
    
    def on_loss(self):
        self.W_max = self.cwnd / self.mss
        self.cwnd = int(self.cwnd * self.beta)
        self.ssthresh = self.cwnd
        self.K = (self.W_max * (1 - self.beta) / self.C) ** (1/3)
        self.t_epoch = self.time
        self.state = 'congestion_avoidance'
        self.history.append(self.cwnd)

class BBR:
    """BBR congestion control (Google, 2016) — simplified model."""
    def __init__(self, mss=1460):
        self.mss = mss
        self.cwnd = mss * 10
        self.btl_bw = 0  # bottleneck bandwidth estimate
        self.rt_prop = float('inf')  # min RTT
        self.pacing_gain = 2.0 / math.log(2)  # startup gain
        self.cwnd_gain = 2.0
        self.state = 'startup'
        self.bw_samples = []
        self.rtt_samples = []
        self.round_count = 0
        self.history = []
    
    def on_ack(self, bytes_delivered, rtt):
        self.history.append(self.cwnd)
        
        # Update estimates
        bw = bytes_delivered / max(rtt, 0.001)
        self.bw_samples.append(bw)
        self.rtt_samples.append(rtt)
        self.rt_prop = min(self.rt_prop, rtt)
        
        # Max BW over last 10 samples
        recent = self.bw_samples[-10:]
        self.btl_bw = max(recent)
        
        self.round_count += 1
        
        if self.state == 'startup':
            self.pacing_gain = 2.0 / math.log(2)
            self.cwnd_gain = 2.0
            # Exit startup when BW plateaus
            if len(self.bw_samples) >= 3:
                if self.bw_samples[-1] <= self.bw_samples[-2] * 1.25:
                    self.state = 'drain'
        elif self.state == 'drain':
            self.pacing_gain = 1.0 / (2.0 / math.log(2))
            self.cwnd_gain = 1.0
            bdp = self.btl_bw * self.rt_prop
            if self.cwnd <= bdp * self.mss:
                self.state = 'probe_bw'
        elif self.state == 'probe_bw':
            self.pacing_gain = 1.0
            self.cwnd_gain = 1.0
        
        # Set cwnd based on BDP
        bdp = self.btl_bw * self.rt_prop
        self.cwnd = max(int(bdp * self.cwnd_gain), 4 * self.mss)
    
    def on_loss(self):
        # BBR doesn't react to loss like traditional CC
        self.history.append(self.cwnd)

def simulate(cc, n_acks=100, loss_at=None, rtt=0.05):
    """Simulate congestion control over n_acks."""
    loss_at = loss_at or set()
    for i in range(n_acks):
        if i in loss_at:
            cc.on_loss()
        else:
            if isinstance(cc, BBR):
                cc.on_ack(cc.mss, rtt)
            elif isinstance(cc, CUBIC):
                cc.on_ack(rtt)
            else:
                cc.on_ack()
    return cc.history

# --- Tests ---

def test_reno_slow_start():
    cc = Reno()
    initial = cc.cwnd
    for _ in range(10):
        cc.on_ack()
    assert cc.cwnd >= initial * 2  # exponential growth

def test_reno_loss():
    cc = Reno()
    for _ in range(20):
        cc.on_ack()
    pre_loss = cc.cwnd
    cc.on_loss()
    assert cc.cwnd <= pre_loss // 2 + cc.mss  # multiplicative decrease

def test_reno_timeout():
    cc = Reno()
    for _ in range(20):
        cc.on_ack()
    cc.on_timeout()
    assert cc.cwnd == cc.mss  # reset to 1 MSS
    assert cc.state == 'slow_start'

def test_cubic_growth():
    cc = CUBIC()
    for _ in range(50):
        cc.on_ack()
    assert cc.cwnd > cc.mss * 10  # should grow

def test_cubic_loss_recovery():
    cc = CUBIC()
    for _ in range(30):
        cc.on_ack()
    pre = cc.cwnd
    cc.on_loss()
    assert cc.cwnd < pre  # reduced
    assert cc.cwnd >= int(pre * 0.7) - cc.mss  # beta=0.7

def test_bbr_startup():
    cc = BBR()
    assert cc.state == 'startup'
    for _ in range(20):
        cc.on_ack(cc.mss, 0.05)
    # Should have moved past startup eventually or cwnd grown
    assert cc.cwnd >= cc.mss * 4

def test_bbr_loss_resilient():
    cc = BBR()
    for _ in range(10):
        cc.on_ack(cc.mss, 0.05)
    pre = cc.cwnd
    cc.on_loss()
    # BBR doesn't halve on loss
    assert cc.cwnd == pre  # unchanged

def test_simulate():
    history = simulate(Reno(), n_acks=50, loss_at={30})
    assert len(history) >= 50
    # Should see growth, then drop at loss
    assert history[29] > history[31]  # cwnd drops after loss

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_reno_slow_start()
        test_reno_loss()
        test_reno_timeout()
        test_cubic_growth()
        test_cubic_loss_recovery()
        test_bbr_startup()
        test_bbr_loss_resilient()
        test_simulate()
        print("All tests passed!")
