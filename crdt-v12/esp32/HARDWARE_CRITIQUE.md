# Hardware Implementation — Critique & Gap Analysis

**Author:** Aaditya Binod Yadav

---

## What the ESP32 port does correctly

| Aspect | Detail |
|---|---|
| HLC correctness | `tick()` and `merge()` preserve the pt > lc tie-breaking invariant |
| OR-Set merge | Idempotent, commutative `add()` with LWW fallback on duplicate IDs |
| Wire format | Identical byte layout to `simulator/wire.py` — interoperable |
| ESP-NOW framing | Stays within 250-byte MTU; no heap allocation in hot path |
| Neighbour eviction | Stale entries pruned after 30 s (matches simulator timeout) |
| Anti-entropy | State hash comparison → DELTA flood on mismatch |

---

## Gaps vs the Python simulation

### 1. RSSI is not available in the ESP-NOW receive callback

`on_data_recv` receives `(mac, data, len)` — no RSSI field.  
**Fix:** Call `esp_wifi_80211_tx` in promiscuous mode (sniffer) or
read the last-packet RSSI from `esp_wifi_sta_get_ap_info` if an AP is
present.  For pure peer-to-peer ESP-NOW there is no official RSSI API;
workaround is to encode RSSI in the HELLO payload (the sender reads its
own Tx power and the receiver can't verify it).

**Impact on ML features:** F2 (rssi_norm) and F11 (mobility proxy) are
unavailable without RSSI.  The 12-feature classifier degrades to 10
usable features on real hardware.

### 2. No persistent storage

OR-Set resets on power cycle.  A crash-safe write to SPIFFS/LittleFS
every N messages would provide durability.  Not implemented here to keep
the sketch self-contained.

### 3. JSON parsing is brittle

`strstr` + `atol` parsing breaks on: nested braces, values before
`"o":`, or payloads with escaped quotes.  A production build should use
ArduinoJson (v6, stack-allocated `StaticJsonDocument<256>`).

### 4. Flooding only — no ML or AODV relay selection

`forward_message` broadcasts to all neighbours.  The `AIStrategy` and
`AODVStrategy` from the simulator are **not** ported here because:
- AIStrategy requires the 12-feature vector, which needs RSSI (gap 1)
  and delivery history state across packets
- AODVStrategy requires selecting one neighbour by MAC, which needs the
  neighbour table to be populated first (fine) but also reliable RSSI

To add AODV on hardware: replace `esp_now_send(BROADCAST_MAC, ...)` in
`forward_message` with `esp_now_send(best_neighbour_mac, ...)` where
`best_neighbour_mac` is chosen by link score from the neighbour table.

### 5. No encryption or authentication

ESP-NOW supports AES-128 CCM encryption via `peer.encrypt = true` and
`esp_now_set_pmk()`.  Not enabled here.  A production deployment must
enable it to prevent message injection.

### 6. No LoRa fallback

The sketch targets ESP-NOW (WiFi, ~100 m range).  For the disaster
scenario (collapsed buildings, multi-km range) an SX1276 LoRa module
should be added on SPI.  The wire format is radio-agnostic; only the
`esp_now_send` / `on_data_recv` functions need to be swapped for
`LoRa.beginPacket` / `LoRa.onReceive`.

---

## Flash / RAM estimates (ESP32-WROOM-32)

| Component | Flash | RAM |
|---|---|---|
| Arduino core + WiFi | ~120 KB | ~30 KB |
| OR-Set (32 msgs × ~80 B) | — | ~2.6 KB |
| Neighbour table (10 × ~24 B) | — | ~0.24 KB |
| Wire codec | ~1.2 KB | stack only |
| **Total estimate** | **~125 KB** | **~35 KB** |

ESP32-WROOM-32 has 4 MB flash and 520 KB SRAM — headroom is ample.

---

## Recommended next steps

1. Add `ArduinoJson` for safe payload parsing
2. Add SPIFFS persistence: write OR-Set to `/orset.bin` on every add
3. Add RSSI via promiscuous-mode sniffer or encode in HELLO payload
4. Implement selective forwarding (AODV unicast vs flood) as a `#define`
5. Add LoRa radio support (swap send/recv layer only — protocol unchanged)
6. Enable ESP-NOW encryption with a pre-shared mesh key
