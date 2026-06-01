"""
Wireless Channel Model for Infrastructure-less Mesh Networks.

Implements a realistic radio propagation model combining:
1. Log-distance path loss (Rappaport, 2002)
2. Log-normal shadowing (Gaussian fading in dB domain)
3. SINR-based packet error rate estimation
4. RSSI computation

Parameters are calibrated for 2.4 GHz ISM band (IEEE 802.11g / ESP-NOW),
typical of resource-constrained IoT devices.

Reference:
    Rappaport, T.S. (2002). Wireless Communications: Principles and
    Practice (2nd ed.). Prentice Hall.

    IEEE 802.11-2020. IEEE Standard for Information Technology —
    Telecommunications and Information Exchange Between Systems.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ChannelConfig:
    """
    Wireless channel configuration parameters.

    Based on IEEE 802.11g at 2.4 GHz for ESP32-class devices.
    """
    # Transmit power (dBm) — ESP32 typical: 18-20 dBm
    tx_power_dbm: float = 20.0

    # Frequency (Hz) — 2.4 GHz ISM band
    frequency_hz: float = 2.4e9

    # Path loss exponent
    # 2.0 = free space, 2.7-3.5 = urban, 4.0-6.0 = indoor obstructed
    path_loss_exponent: float = 3.0

    # Reference distance (meters) for path loss model
    reference_distance_m: float = 1.0

    # Path loss at reference distance (dB)
    # For 2.4 GHz: PL(1m) ≈ 40.05 dB (free-space Friis)
    reference_loss_db: float = 40.05

    # Log-normal shadowing standard deviation (dB)
    # 0 = no shadowing, 4-8 = typical urban/indoor
    shadowing_std_db: float = 6.0

    # Noise floor (dBm) — thermal noise + receiver noise figure
    # kTB + NF ≈ -174 + 10*log10(BW) + NF
    # For 20 MHz BW, NF=10dB: -174 + 73 + 10 = -91 dBm
    noise_floor_dbm: float = -91.0

    # Minimum SINR for successful reception (dB)
    # For 6 Mbps OFDM (BPSK 1/2): ~6 dB
    # For 250 kbps ESP-NOW: ~10 dB
    sinr_threshold_db: float = 10.0

    # Data rate (bits per second) — ESP-NOW effective
    data_rate_bps: float = 250_000.0

    # Maximum transmission range (meters) — hard cutoff
    max_range_m: float = 200.0

    # Packet header overhead (bytes) — MAC + PHY headers
    header_overhead_bytes: int = 36


class WirelessChannel:
    """
    Wireless channel simulator with log-distance path loss and shadowing.

    Models the physical-layer behavior of IEEE 802.11g ad-hoc links
    between constrained wireless nodes.

    The channel quality between two nodes depends on:
    1. Distance (deterministic path loss)
    2. Environment (random shadowing/fading)
    3. Packet size (affects bit error accumulation)

    Usage:
        channel = WirelessChannel(ChannelConfig())
        success, rssi, per = channel.attempt_delivery(
            sender_pos=(0, 0), receiver_pos=(50, 30), packet_size=64
        )
    """

    def __init__(self, config: ChannelConfig, rng: np.random.Generator = None):
        self.config = config
        self.rng = rng or np.random.default_rng(42)

        # Precompute wavelength for Friis equation
        c = 3e8  # speed of light (m/s)
        self._wavelength = c / config.frequency_hz

    def distance(self, pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
        """Euclidean distance between two positions."""
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        return math.sqrt(dx * dx + dy * dy)

    def path_loss(self, distance_m: float) -> float:
        """
        Log-distance path loss model (dB).

        PL(d) = PL(d₀) + 10·n·log₁₀(d/d₀) + X_σ

        where:
            PL(d₀) = reference loss at d₀
            n = path loss exponent
            X_σ = zero-mean Gaussian (log-normal shadowing)

        Reference: Rappaport (2002), Eq. 4.69.
        """
        if distance_m <= 0:
            return 0.0

        d = max(distance_m, self.config.reference_distance_m)
        cfg = self.config

        # Deterministic path loss
        pl = cfg.reference_loss_db + 10.0 * cfg.path_loss_exponent * math.log10(
            d / cfg.reference_distance_m
        )

        # Stochastic shadowing component
        if cfg.shadowing_std_db > 0:
            shadow = self.rng.normal(0, cfg.shadowing_std_db)
            pl += shadow

        return pl

    def rssi(self, distance_m: float) -> float:
        """
        Received Signal Strength Indicator (dBm).

        RSSI = Tx_Power - PathLoss(d)
        """
        pl = self.path_loss(distance_m)
        return self.config.tx_power_dbm - pl

    def sinr(self, rssi_dbm: float) -> float:
        """
        Signal-to-Interference-plus-Noise Ratio (dB).

        SINR = RSSI - NoiseFloor
        (simplified: no co-channel interference modeled)
        """
        return rssi_dbm - self.config.noise_floor_dbm

    def packet_error_rate(self, sinr_db: float, packet_size_bytes: int) -> float:
        """
        Estimate Packet Error Rate (PER) from SINR.

        Uses a sigmoid approximation of the BER-to-PER relationship:
            BER ≈ 0.5 * erfc(sqrt(SINR_linear / 2))
            PER = 1 - (1 - BER)^(8 * packet_size)

        For computational efficiency, we use a sigmoid approximation:
            PER ≈ 1 / (1 + exp(k * (SINR - threshold)))

        Reference:
            Haccoun, D. & Begin, G. (1989). High-rate punctured
            convolutional codes for soft decision Viterbi decoding.
            IEEE Trans. on Communications.
        """
        # Sigmoid approximation parameters
        k = 1.2  # steepness of PER curve
        threshold = self.config.sinr_threshold_db

        # PER = 1 / (1 + exp(k * (SINR - threshold)))
        exponent = k * (sinr_db - threshold)
        exponent = max(-50, min(50, exponent))  # clamp to avoid overflow

        per = 1.0 / (1.0 + math.exp(exponent))

        # Scale PER by packet size (larger packets more likely to have errors)
        # Approximate: PER_packet ≈ 1 - (1 - PER_symbol)^(bits)
        bits = packet_size_bytes * 8
        reference_bits = 256 * 8  # reference packet size
        size_factor = bits / reference_bits
        per = 1.0 - (1.0 - per) ** size_factor

        return max(0.0, min(1.0, per))

    def attempt_delivery(
        self,
        sender_pos: Tuple[float, float],
        receiver_pos: Tuple[float, float],
        packet_size_bytes: int = 64,
    ) -> Tuple[bool, float, float]:
        """
        Simulate a single packet transmission attempt.

        Args:
            sender_pos: (x, y) position of sender in meters.
            receiver_pos: (x, y) position of receiver in meters.
            packet_size_bytes: Total packet size including headers.

        Returns:
            Tuple of (success, rssi_dbm, packet_error_rate):
                success: True if packet was delivered successfully.
                rssi_dbm: Received signal strength at receiver.
                per: Calculated packet error rate.
        """
        dist = self.distance(sender_pos, receiver_pos)

        # Hard range cutoff
        if dist > self.config.max_range_m:
            return (False, -120.0, 1.0)

        # Compute RSSI and SINR
        total_size = packet_size_bytes + self.config.header_overhead_bytes
        rssi_dbm = self.rssi(dist)
        sinr_db = self.sinr(rssi_dbm)
        per = self.packet_error_rate(sinr_db, total_size)

        # Bernoulli trial for delivery success
        success = self.rng.random() > per

        return (success, rssi_dbm, per)

    def propagation_delay(self, distance_m: float) -> float:
        """
        Propagation + transmission delay (seconds).

        Includes:
        - Radio propagation (~3.3 ns/m, negligible)
        - Processing delay (~1 ms per hop)
        - Random jitter (0-5 ms)
        """
        prop = distance_m / 3e8  # speed of light
        processing = 0.001  # 1 ms processing
        jitter = self.rng.uniform(0, 0.005)  # 0-5 ms jitter
        return prop + processing + jitter

    def transmission_time(self, packet_size_bytes: int) -> float:
        """
        Time to transmit packet at the configured data rate.

        Returns time in seconds.
        """
        bits = (packet_size_bytes + self.config.header_overhead_bytes) * 8
        return bits / self.config.data_rate_bps

    def effective_range(self, target_per: float = 0.1) -> float:
        """
        Compute the effective range for a given target PER.

        Returns the maximum distance at which PER ≤ target_per
        (without shadowing).
        """
        cfg = self.config

        # Required SINR for target PER
        # PER = 1 / (1 + exp(k * (SINR - threshold)))
        # Solving: SINR = threshold + ln(1/PER - 1) / k
        k = 1.2
        if target_per <= 0 or target_per >= 1:
            return 0.0
        required_sinr = cfg.sinr_threshold_db + math.log(1.0 / target_per - 1.0) / k

        # Required RSSI
        required_rssi = required_sinr + cfg.noise_floor_dbm

        # Required path loss
        max_pl = cfg.tx_power_dbm - required_rssi

        # Solve for distance: PL(d) = PL(d0) + 10*n*log10(d/d0)
        if max_pl <= cfg.reference_loss_db:
            return cfg.reference_distance_m

        exponent = (max_pl - cfg.reference_loss_db) / (10.0 * cfg.path_loss_exponent)
        distance = cfg.reference_distance_m * (10.0 ** exponent)

        return min(distance, cfg.max_range_m)

    def __repr__(self) -> str:
        eff_range = self.effective_range(0.1)
        return (
            f"WirelessChannel(tx={self.config.tx_power_dbm}dBm, "
            f"n={self.config.path_loss_exponent}, "
            f"σ={self.config.shadowing_std_db}dB, "
            f"eff_range≈{eff_range:.0f}m)"
        )
