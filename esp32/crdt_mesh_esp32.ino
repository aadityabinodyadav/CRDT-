/**
 * CRDT Mesh — ESP32 / ESP-NOW Hardware Implementation
 *
 * Ports the Python simulation to C++ for ESP32 boards communicating
 * via ESP-NOW (WiFi peer-to-peer, no AP required).  Designed for
 * disaster-response mesh where infrastructure is absent.
 *
 * Hardware target:
 *   ESP32-WROOM-32 (or ESP32-S3) + 2.4 GHz on-board antenna
 *   Optional: SX1276 LoRa via SPI on pins 5/18/19/27 for 915 MHz long-range
 *
 * Flash: ~180 KB  |  RAM: ~45 KB peak  |  Power: see EnergyMode enum
 *
 * Dependencies (install via Arduino Library Manager):
 *   - ESP32 Arduino core >= 2.0.14
 *   - (optional) LoRa by Sandeep Mistry, for SX1276 radio
 *
 * Protocol summary:
 *   1. Boot → generate unique node_id from MAC
 *   2. Broadcast HELLO every HELLO_INTERVAL_MS
 *   3. On receiving HELLO → update neighbour table
 *   4. When a message is generated → wrap in OR-Set entry, gossip to neighbours
 *   5. On receiving DATA → HLC merge, OR-Set merge, forward if unseen
 *   6. Every SYNC_INTERVAL_MS → anti-entropy: exchange state hashes, DELTA if mismatch
 *
 * Wire format: length-prefixed binary (matches simulator/wire.py)
 *   [2B total_len][1B type][4B sender_id][8B ts_ms (double)][2B payload_len][N B payload]
 *
 * Author: Aaditya Binod Yadav
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <string.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

#define HELLO_INTERVAL_MS      3000
#define SYNC_INTERVAL_MS       5000
#define MAX_NEIGHBOURS         10
#define MAX_MESSAGES           32
#define MSG_TTL_HOPS           8
#define NEIGHBOUR_TIMEOUT_MS   30000
#define MAX_PAYLOAD_BYTES      240   // ESP-NOW max is 250 bytes total

#define PKT_HELLO    0
#define PKT_DATA     1
#define PKT_SUMMARY  2
#define PKT_DELTA    3

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid Logical Clock
// ─────────────────────────────────────────────────────────────────────────────

struct HLC {
  uint64_t pt;   // physical time ms
  uint32_t lc;   // logical counter

  void tick() {
    uint64_t now_ms = (uint64_t)millis();
    if (now_ms > pt) { pt = now_ms; lc = 0; }
    else              { lc++; }
  }

  void merge(const HLC& o) {
    if (o.pt > pt || (o.pt == pt && o.lc > lc)) { pt = o.pt; lc = o.lc; }
    lc++;
  }

  bool operator<(const HLC& o) const {
    return (pt < o.pt) || (pt == o.pt && lc < o.lc);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Message / OR-Set
// ─────────────────────────────────────────────────────────────────────────────

struct MessageId { uint32_t origin; uint32_t seq; };

struct Message {
  MessageId  id;
  HLC        timestamp;
  uint8_t    hop_count;
  char       payload[64];
  bool       active;
};

class ORSet {
public:
  Message entries[MAX_MESSAGES];
  uint8_t count = 0;

  int find(const MessageId& id) const {
    for (int i = 0; i < count; i++)
      if (entries[i].id.origin == id.origin && entries[i].id.seq == id.seq)
        return i;
    return -1;
  }

  bool add(const Message& msg) {
    int idx = find(msg.id);
    if (idx >= 0) {
      if (entries[idx].timestamp < msg.timestamp) entries[idx] = msg;
      return false;
    }
    if (count >= MAX_MESSAGES) _evict_oldest();
    entries[count++] = msg;
    return true;
  }

  void merge(const ORSet& other) {
    for (int i = 0; i < other.count; i++) add(other.entries[i]);
  }

  uint32_t state_hash() const {
    uint32_t h = 0;
    for (int i = 0; i < count; i++)
      if (entries[i].active)
        h ^= (entries[i].id.origin * 2654435761u) ^ entries[i].id.seq;
    return h;
  }

private:
  void _evict_oldest() {
    int oldest = 0;
    for (int i = 1; i < count; i++)
      if (entries[i].timestamp < entries[oldest].timestamp) oldest = i;
    for (int i = oldest; i < count - 1; i++) entries[i] = entries[i+1];
    count--;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Neighbour table
// ─────────────────────────────────────────────────────────────────────────────

struct NeighbourInfo {
  uint8_t  mac[6];
  uint32_t node_id;
  uint32_t last_seen_ms;
  int8_t   rssi;
  uint8_t  degree;
  uint32_t state_hash;
  bool     active;
};

// ─────────────────────────────────────────────────────────────────────────────
// Wire format (matches simulator/wire.py)
// ─────────────────────────────────────────────────────────────────────────────

uint16_t wire_pack(uint8_t* buf, uint16_t buf_len,
                   uint8_t pkt_type, uint32_t sender_id,
                   double ts_s, const char* json) {
  uint16_t plen  = (uint16_t)strlen(json);
  uint16_t total = 2 + 1 + 4 + 8 + 2 + plen;
  if (total > buf_len) return 0;

  uint16_t off = 0;
  buf[off++] = (total >> 8) & 0xFF; buf[off++] = total & 0xFF;
  buf[off++] = pkt_type;
  buf[off++] = (sender_id >> 24) & 0xFF; buf[off++] = (sender_id >> 16) & 0xFF;
  buf[off++] = (sender_id >>  8) & 0xFF; buf[off++] =  sender_id        & 0xFF;

  uint8_t ts_b[8]; memcpy(ts_b, &ts_s, 8);
  // little-endian ESP32 → big-endian wire
  for (int i = 0; i < 4; i++) { uint8_t t=ts_b[i]; ts_b[i]=ts_b[7-i]; ts_b[7-i]=t; }
  memcpy(buf+off, ts_b, 8); off += 8;

  buf[off++] = (plen >> 8) & 0xFF; buf[off++] = plen & 0xFF;
  memcpy(buf+off, json, plen);
  return total;
}

struct WirePacket {
  uint8_t  pkt_type;
  uint32_t sender_id;
  char     payload[MAX_PAYLOAD_BYTES];
  uint16_t payload_len;
  bool     valid;
};

WirePacket wire_unpack(const uint8_t* buf, uint16_t buf_len) {
  WirePacket out = {};
  if (buf_len < 17) return out;
  uint16_t off = 0;
  uint16_t total_len = ((uint16_t)buf[0]<<8)|buf[1]; off+=2;
  if (total_len > buf_len) return out;
  out.pkt_type  = buf[off++];
  out.sender_id = ((uint32_t)buf[off]<<24)|((uint32_t)buf[off+1]<<16)|
                  ((uint32_t)buf[off+2]<<8)|(uint32_t)buf[off+3]; off+=4;
  off += 8;  // skip timestamp
  uint16_t plen = ((uint16_t)buf[off]<<8)|buf[off+1]; off+=2;
  if (plen >= MAX_PAYLOAD_BYTES || off+plen > buf_len) return out;
  memcpy(out.payload, buf+off, plen);
  out.payload[plen] = '\0';
  out.payload_len = plen;
  out.valid = true;
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Global state
// ─────────────────────────────────────────────────────────────────────────────

static uint32_t      g_node_id;
static uint8_t       g_mac[6];
static HLC           g_hlc;
static ORSet         g_orset;
static NeighbourInfo g_neighbours[MAX_NEIGHBOURS];
static uint8_t       g_nbr_count = 0;
static uint32_t      g_seq = 0;
static uint32_t      g_last_hello_ms = 0;
static uint32_t      g_last_sync_ms  = 0;
static uint32_t      g_last_gen_ms   = 0;
static uint8_t       BROADCAST_MAC[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

// ─────────────────────────────────────────────────────────────────────────────
// Neighbour helpers
// ─────────────────────────────────────────────────────────────────────────────

NeighbourInfo* find_neighbour(uint32_t node_id) {
  for (int i = 0; i < g_nbr_count; i++)
    if (g_neighbours[i].node_id == node_id && g_neighbours[i].active)
      return &g_neighbours[i];
  return nullptr;
}

NeighbourInfo* add_neighbour(uint32_t node_id, const uint8_t* mac) {
  NeighbourInfo* ex = find_neighbour(node_id);
  if (ex) return ex;
  for (int i = 0; i < MAX_NEIGHBOURS; i++) {
    if (!g_neighbours[i].active) {
      memset(&g_neighbours[i], 0, sizeof(NeighbourInfo));
      g_neighbours[i].node_id = node_id;
      memcpy(g_neighbours[i].mac, mac, 6);
      g_neighbours[i].active = true;
      if (i >= g_nbr_count) g_nbr_count = i+1;
      esp_now_peer_info_t peer = {};
      memcpy(peer.peer_addr, mac, 6);
      peer.channel = 0; peer.encrypt = false;
      esp_now_add_peer(&peer);
      return &g_neighbours[i];
    }
  }
  return nullptr;
}

void prune_stale_neighbours() {
  uint32_t now = millis();
  for (int i = 0; i < g_nbr_count; i++)
    if (g_neighbours[i].active && now - g_neighbours[i].last_seen_ms > NEIGHBOUR_TIMEOUT_MS)
      g_neighbours[i].active = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Send
// ─────────────────────────────────────────────────────────────────────────────

void send_hello() {
  char json[96];
  int active = 0;
  for (int i = 0; i < g_nbr_count; i++) if (g_neighbours[i].active) active++;
  snprintf(json, sizeof(json), "{\"h\":%lu,\"d\":%d,\"m\":%d}",
           (unsigned long)g_orset.state_hash(), active, g_orset.count);
  uint8_t frame[MAX_PAYLOAD_BYTES];
  uint16_t len = wire_pack(frame, sizeof(frame), PKT_HELLO, g_node_id, millis()/1000.0, json);
  if (len) esp_now_send(BROADCAST_MAC, frame, len);
}

void forward_message(const Message& msg) {
  if (msg.hop_count >= MSG_TTL_HOPS) return;
  Message fwd = msg; fwd.hop_count++;
  char json[MAX_PAYLOAD_BYTES-20];
  snprintf(json, sizeof(json),
    "{\"o\":%lu,\"s\":%lu,\"h\":%d,\"p\":\"%s\",\"pt\":%llu,\"lc\":%lu}",
    (unsigned long)fwd.id.origin, (unsigned long)fwd.id.seq,
    fwd.hop_count, fwd.payload,
    (unsigned long long)fwd.timestamp.pt, (unsigned long)fwd.timestamp.lc);
  uint8_t frame[MAX_PAYLOAD_BYTES];
  uint16_t len = wire_pack(frame, sizeof(frame), PKT_DATA, g_node_id, millis()/1000.0, json);
  if (len) esp_now_send(BROADCAST_MAC, frame, len);
}

// ─────────────────────────────────────────────────────────────────────────────
// ESP-NOW callbacks
// ─────────────────────────────────────────────────────────────────────────────

void on_data_recv(const uint8_t* mac, const uint8_t* data, int len) {
  if (len < 17 || len > MAX_PAYLOAD_BYTES) return;
  WirePacket pkt = wire_unpack(data, (uint16_t)len);
  if (!pkt.valid || pkt.sender_id == g_node_id) return;

  NeighbourInfo* nbr = find_neighbour(pkt.sender_id);
  if (!nbr) nbr = add_neighbour(pkt.sender_id, mac);
  if (nbr) nbr->last_seen_ms = millis();

  if (pkt.pkt_type == PKT_HELLO) {
    if (nbr) {
      char* dp = strstr(pkt.payload, "\"d\":"); if (dp) nbr->degree = (uint8_t)atoi(dp+4);
      char* hp = strstr(pkt.payload, "\"h\":"); if (hp) nbr->state_hash = (uint32_t)atol(hp+4);
    }
  } else if (pkt.pkt_type == PKT_DATA) {
    char* op = strstr(pkt.payload, "\"o\":"); if (!op) return;
    char* sp = strstr(pkt.payload, "\"s\":"); if (!sp) return;
    Message msg = {}; msg.active = true;
    msg.id.origin = (uint32_t)atol(op+4);
    msg.id.seq    = (uint32_t)atol(sp+4);
    char* hp = strstr(pkt.payload, "\"h\":"); if (hp) msg.hop_count = (uint8_t)atoi(hp+4);
    char* pt = strstr(pkt.payload, "\"pt\":"); if (pt) msg.timestamp.pt = (uint64_t)atoll(pt+5);
    char* lc = strstr(pkt.payload, "\"lc\":"); if (lc) msg.timestamp.lc = (uint32_t)atol(lc+5);
    char* pp = strstr(pkt.payload, "\"p\":\"");
    if (pp) {
      char* end = strchr(pp+5, '"');
      if (end) { int n=end-(pp+5); if(n>63)n=63; memcpy(msg.payload,pp+5,n); msg.payload[n]=0; }
    }
    g_hlc.merge(msg.timestamp);
    bool is_new = g_orset.add(msg);
    if (is_new) {
      Serial.printf("[RECV] %lu:%lu hop=%d: %s\n",
        (unsigned long)msg.id.origin,(unsigned long)msg.id.seq,msg.hop_count,msg.payload);
      forward_message(msg);
    }
  } else if (pkt.pkt_type == PKT_SUMMARY) {
    char* hp = strstr(pkt.payload, "\"h\":");
    if (hp && nbr) {
      uint32_t their_hash = (uint32_t)atol(hp+4);
      if (their_hash != g_orset.state_hash())
        for (int i = 0; i < g_orset.count; i++)
          if (g_orset.entries[i].active) forward_message(g_orset.entries[i]);
    }
  }
}

void on_data_sent(const uint8_t* mac, esp_now_send_status_t status) {
  (void)mac; (void)status;
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup / Loop
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200); delay(500);
  WiFi.mode(WIFI_STA);
  esp_read_mac(g_mac, ESP_MAC_WIFI_STA);
  g_node_id = ((uint32_t)g_mac[2]<<24)|((uint32_t)g_mac[3]<<16)|
              ((uint32_t)g_mac[4]<<8)|(uint32_t)g_mac[5];

  Serial.printf("\n=== CRDT Mesh Node ===\nID: %08lX  MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
    (unsigned long)g_node_id,
    g_mac[0],g_mac[1],g_mac[2],g_mac[3],g_mac[4],g_mac[5]);

  if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW FAILED"); while(1) delay(1000); }
  esp_now_register_recv_cb(on_data_recv);
  esp_now_register_send_cb(on_data_sent);
  esp_now_peer_info_t bc={}; memcpy(bc.peer_addr,BROADCAST_MAC,6); bc.encrypt=false;
  esp_now_add_peer(&bc);
  Serial.println("Ready.\n");
}

void loop() {
  uint32_t now = millis();

  if (now - g_last_hello_ms >= HELLO_INTERVAL_MS) {
    g_last_hello_ms = now; send_hello(); prune_stale_neighbours();
  }

  if (now - g_last_sync_ms >= SYNC_INTERVAL_MS) {
    g_last_sync_ms = now;
    char json[48]; snprintf(json,sizeof(json),"{\"h\":%lu}",(unsigned long)g_orset.state_hash());
    uint8_t frame[MAX_PAYLOAD_BYTES];
    uint16_t len = wire_pack(frame,sizeof(frame),PKT_SUMMARY,g_node_id,now/1000.0,json);
    if (len) esp_now_send(BROADCAST_MAC,frame,len);
  }

  if (now - g_last_gen_ms >= 10000) {
    g_last_gen_ms = now;
    Message msg = {}; msg.id.origin=g_node_id; msg.id.seq=++g_seq; msg.active=true;
    g_hlc.tick(); msg.timestamp=g_hlc;
    snprintf(msg.payload,sizeof(msg.payload),"node%lu:msg%lu",
             (unsigned long)g_node_id,(unsigned long)g_seq);
    if (g_orset.add(msg)) { Serial.printf("[GEN] %s\n",msg.payload); forward_message(msg); }
  }

  delay(10);
}
