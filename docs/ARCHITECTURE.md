---
title: Airport Queue Intelligence — System Architecture
version: 1.1
date: 2026-03-30
classification: Confidential — For Airport Technical Team & Integration Partners
---

# Airport Queue Intelligence: System Architecture

This document describes the technical architecture for deploying Airport Queue Intelligence at production scale: ~400 cameras, 15-20 edge devices, one cloud application server.

For product context (what the system does, who uses it, what metrics it produces, alerting, privacy), see [Product Requirements](PRODUCT_REQUIREMENTS.md).

**Short version**: Existing airport CCTV feeds are processed by on-premise edge devices running computer vision. Edge devices detect and track people in queue/service zones, compute wait times and throughput, and send numerical metadata to the application server. The application server runs in the airport's own data center or private cloud. It stores metrics, generates predictions, serves a real-time dashboard to operations staff, and handles edge fleet orchestration. All data stays within the airport network. No video leaves the edge devices.

---

## How to Read This Document

This document is written for the airport IT team and integration partners who need to understand, deploy, and maintain the Airport Queue Intelligence system. It covers the full stack from edge hardware through cloud services to the operator dashboard, including data models, message schemas, database schemas, failover procedures, and operational runbooks. Sections are numbered and ordered from high-level topology down to detailed operations, so you can read end-to-end for a complete picture or jump to a specific section using the table of contents.

---

## Technology Summary

| Technology | Role | Why This Choice |
|---|---|---|
| **FastAPI** | Web framework + REST API + WebSocket gateway | Async I/O handles 800 msgs/s + WebSocket + REST in one process; same language as edge code |
| **TimescaleDB** | Time-series storage (PostgreSQL 16 + extension) | SQL familiarity, hypertables with auto time-partitioning, 10x compression, continuous aggregates |
| **Redis 7.x** | Real-time state cache + pub/sub | Sub-millisecond reads for latest zone metrics; pub/sub fans out to WebSocket gateway |
| **Mosquitto 2.x** | MQTT broker | Lightweight, stable, handles 15-20 edge clients easily; upgrade path to EMQX for clustering |
| **Caddy 2.x** | Reverse proxy | Automatic TLS, WebSocket upgrade, static file serving for the React SPA |
| **React** | Dashboard SPA | Virtual scrolling for 800 zones, component-based area grouping and drill-down views |
| **OpenVINO** | NPU inference runtime | Native Intel NPU 4.0 support, INT8 quantization, THROUGHPUT mode for multi-camera batching |
| **ByteTrack** (baseline) | Multi-object tracker | Lightweight (numpy + scipy only), stable track IDs across frames, ~1-2ms per frame. **Note:** A pre-implementation benchmark against BoT-SORT and OC-SORT is required — ByteTrack may fragment IDs in dense stationary queues. |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Model](#2-data-model)
3. [Edge Device](#3-edge-device)
4. [Cloud Application Server](#4-cloud-application-server)
5. [Data Flow](#5-data-flow)
6. [Database Schema](#6-database-schema)
7. [Dashboard & WebSocket](#7-dashboard--websocket)
8. [Failover & Resilience](#8-failover--resilience)
9. [Operations](#9-operations)
10. [Degradation Matrix](#10-degradation-matrix)

---

## 1. System Overview

### 1.1 Topology

```mermaid
graph TD
    subgraph CLOUD["CLOUD APPLICATION SERVER"]
        Mosquitto["Mosquitto<br/>MQTT Broker<br/><i>Receives all edge telemetry</i>"]
        FastAPI["FastAPI<br/>(uvicorn, 4 workers)<br/><i>REST API + WebSocket gateway</i>"]
        TimescaleDB["TimescaleDB<br/>(PostgreSQL + hypertable)<br/><i>Zone metrics time-series</i>"]
        Redis["Redis<br/>(latest state cache + pub/sub)"]

        MQTTConsumer["MQTT Consumer Service<br/>Subscribes to all edge topics.<br/>1. Write latest → Redis hash<br/>2. Publish → Redis pub/sub<br/>3. Batch insert → TimescaleDB"]

        Orchestrator["Orchestrator<br/>health_monitor: watches edge heartbeats<br/>camera_assigner: assigns cameras, handles failover<br/>config_pusher: pushes zone/threshold changes via MQTT"]

        WSGateway["WebSocket Gateway<br/>Subscribes to Redis pub/sub.<br/>Per-browser-client subscription filters.<br/>Pushes only relevant zone/area updates."]

        Caddy["Caddy<br/>(reverse proxy — TLS, static files, WebSocket upgrade)"]

        Mosquitto --> MQTTConsumer
        MQTTConsumer --> Redis
        MQTTConsumer --> TimescaleDB
        Redis --> WSGateway
        FastAPI --> TimescaleDB
        FastAPI --> Redis
        Orchestrator --> Mosquitto
    end

    LAN["Airport LAN / WAN"]
    CLOUD --> LAN

    subgraph EDGE1["Edge-01"]
        E1_iGPU["iGPU decode"]
        E1_NPU["NPU YOLOv8"]
        E1_CPU["CPU ByteTrack + Analytics"]
        E1_MQTT["MQTT publisher"]
        E1_iGPU --> E1_NPU --> E1_CPU --> E1_MQTT
    end

    subgraph EDGE2["Edge-02"]
        E2_Stack["Same stack"]
    end

    subgraph EDGEN["Edge-15"]
        EN_Stack["Same stack"]
    end

    LAN --> EDGE1
    LAN --> EDGE2
    LAN --> EDGEN

    CAM1["30 RTSP Cameras<br/>(network-accessible from any edge)"]
    CAM2["30 RTSP Cameras"]
    CAMN["25 RTSP Cameras"]

    CAM1 --> EDGE1
    CAM2 --> EDGE2
    CAMN --> EDGEN
```

All components run on the airport's own network. The application server sits in the airport's data center or private cloud. Edge devices are in server rooms or comms closets on-premise. No data leaves the airport. Edge devices push metadata only (no video) to the application server.

### 1.2 Scale Numbers

| Dimension | Count |
|---|---|
| RTSP cameras | ~400 |
| Edge devices | 15-20 |
| Cameras per edge | 25-35 |
| Zones per camera (avg) | ~2 (some cameras cover 1, some cover 3-4) |
| Total zones | ~800 |
| Zone metrics messages/sec | ~800 |
| Total metadata bandwidth | ~431 KB/s |

### 1.3 What Runs Where

| Function | Where | Hardware |
|---|---|---|
| Video decode (H.264/H.265) | Edge | Intel Arc 140T iGPU (QSV/VA-API) |
| Person detection (YOLOv8 INT8) | Edge | Intel NPU 4.0 (~48 TOPS) |
| Object tracking (ByteTrack) | Edge | CPU (6P+8E cores) |
| Zone analytics (dwell time) | Edge | CPU |
| MQTT publish | Edge | CPU + network |
| Metadata ingestion | Cloud | CPU |
| Time-series storage | Cloud | TimescaleDB (disk) |
| Real-time state cache | Cloud | Redis (RAM) |
| Dashboard serving | Cloud | FastAPI + WebSocket |
| Orchestration & failover | Cloud | CPU |

---

## 2. Data Model

### 2.1 Zone-Centric Hierarchy

The application's primary entity is the **Zone**, not the camera.

Operators think in "Checkout Counter A", "Security Lane 3", "Gate B4 Queue". A camera is infrastructure -a sensor that feeds one or more zones. A zone's identity, history, alerts, and thresholds persist across camera swaps and edge failovers.

```mermaid
graph TD
    Airport["Airport"]

    AreaA["Area: Terminal A Security"]
    AreaB["Area: Terminal B Check-in"]
    AreaC["Area: Gate C Boarding"]

    Airport --> AreaA
    Airport --> AreaB
    Airport --> AreaC

    ZoneA1["Zone: Lane 1"]
    ZoneA2["Zone: Lane 2"]
    ZoneA3["Zone: Lane 3"]
    ZoneA4["Zone: Lane 4"]
    AreaA --> ZoneA1
    AreaA --> ZoneA2
    AreaA --> ZoneA3
    AreaA --> ZoneA4

    ZoneB1["Zone: Counter A"]
    ZoneB2["Zone: Counter B"]
    ZoneB3["Zone: Counter C"]
    AreaB --> ZoneB1
    AreaB --> ZoneB2
    AreaB --> ZoneB3

    ZoneC1["Zone: Gate C4 Queue"]
    AreaC --> ZoneC1

    CamA1["Camera CAM-A1-01"]
    CamA2["Camera CAM-A1-02"]
    CamB1["Camera CAM-B2-01"]
    CamB2["Camera CAM-B2-02"]
    CamC1["Camera CAM-C4-01"]

    ZoneA1 -->|fed by| CamA1
    ZoneA2 -->|fed by| CamA1
    ZoneA3 -->|fed by| CamA2
    ZoneA4 -->|fed by| CamA2
    ZoneB1 -->|fed by| CamB1
    ZoneB2 -->|fed by| CamB1
    ZoneB3 -->|fed by| CamB2
    ZoneC1 -->|fed by| CamC1

    Edge01["Edge-01"]
    Edge03["Edge-03"]
    Edge07["Edge-07"]

    CamA1 -->|runs on| Edge01
    CamA2 -->|runs on| Edge01
    CamB1 -->|runs on| Edge03
    CamB2 -->|runs on| Edge03
    CamC1 -->|runs on| Edge07
```

One camera can feed multiple zones (e.g., CAM-A1-01 sees both Lane 1 and Lane 2). Inference runs once per frame per camera; each zone runs its own independent `QueueAnalytics` instance against the same detection results. CPU overhead per extra zone is ~0.5ms (polygon point-in-zone checks + debounce + dwell tracking).

### 2.2 Entity Relationships

```mermaid
erDiagram
    areas {
        TEXT area_id PK
        TEXT name
        TEXT terminal
        TEXT category
    }

    zones {
        TEXT zone_id PK
        TEXT name
        TEXT area_id FK
        TEXT camera_id FK
        JSONB queue_poly
        JSONB service_poly
        INT alert_thr
        REAL wait_thr
        REAL conf_thr
    }

    cameras {
        TEXT camera_id PK
        TEXT name
        TEXT source_url
        TEXT assigned_edge FK
        TEXT status
    }

    edges {
        TEXT edge_id PK
        TEXT hostname
        INET ip_address
        INT max_cameras
        TEXT status
    }

    zone_metrics {
        TIMESTAMPTZ ts
        TEXT zone_id FK
        SMALLINT queue_length
        REAL avg_wait
        REAL predicted
        REAL throughput
    }

    areas ||--o{ zones : "1 to N"
    zones }o--|| cameras : "N to 1"
    cameras }o--|| edges : "N to 1"
    zones ||--o{ zone_metrics : "1 to N"
```

### 2.3 Why Zone-Centric

| Concern | Camera-Centric | Zone-Centric |
|---|---|---|
| Camera swap (hardware failure) | History breaks -new camera_id | Zone keeps its history |
| Edge failover | Metrics key changes | zone_id unchanged |
| Dashboard grouping | Operators must know camera->queue mapping | Operators see named queues directly |
| Alerts | "Camera 47 queue > 10" (meaningless) | "Security Lane 3 queue > 10" |
| Multi-queue cameras | Awkward q2_ prefix or compound keys | Each zone is a first-class entity |

---

## 3. Edge Device

### 3.1 Hardware Per Device

| Component | Spec | Role |
|---|---|---|
| CPU | Intel Core Ultra 7 255H (6P+8E cores) | ByteTrack, analytics, MQTT client |
| NPU | Intel AI Boost NPU 4.0 (~48 TOPS) | YOLOv8 INT8 person detection |
| iGPU | Intel Arc 140T | Hardware video decode (QSV/VA-API) |
| RAM | 32 GB DDR5 dual-channel | Shared across CPU/NPU/iGPU |
| OS | Ubuntu 22.04 LTS | Intel NPU drivers + OpenVINO |

### 3.2 Software Components

The current monolith (`flow-line-single-zone.py`, 1842 lines) is refactored into a **headless edge agent** -no Flask, no MJPEG, no dashboard.

```
edge_agent/
  main.py
    # Entry point. Connects MQTT, loads assignment,
    # starts pipeline.

  pipeline/
    video_decoder.py
      # VideoDecoder class. iGPU decode:
      # GStreamer VA-API → FFmpeg QSV → CPU fallback.
      # One instance per camera. Threaded capture loop.

    npu_inference.py
      # NPUInferenceEngine class.
      # 8 worker threads, shared frame queue.
      # OpenVINO THROUGHPUT mode, INT8 YOLOv8 on NPU.
      # One engine per device (singleton).

    bytetrack.py
      # Multi-object tracker (~350 lines, numpy+scipy).
      # Default: ByteTrack.
      # Alternatives: BoT-SORT, OC-SORT.
      # NOTE: A tracker benchmark on airport footage
      # must run before finalizing the choice.
      # ByteTrack assumes linear motion; dense queues
      # may cause excessive ID switches.
      # One tracker instance per camera.
      # Kalman filter + Hungarian matching.

    queue_analytics.py
      # QueueAnalytics class (dwell-time based).
      # One instance per zone (N per camera).
      # Tracks zone entry/exit per track_id.
      # Computes: queue_length, avg_wait,
      # predicted_wait, p50/p90, throughput, alerts.

    camera_stream.py
      # CameraStream class.
      # Manages: VideoDecoder + Tracker + Analytics.
      # Submit loop: 5 FPS → NPU engine.
      # Callback: tracker → analytics → metrics.
      # No drawing, no MJPEG, no Flask.

  comms/
    mqtt_publisher.py
      # Publishes zone metrics, heartbeats,
      # checkpoints, alerts.
      # Uses paho-mqtt with SQLite offline buffer.

    mqtt_listener.py
      # Subscribes to control topics:
      # - Camera assign/remove commands
      # - Zone config updates (polygon, thresholds)
      # - Broadcast commands (stop_all, restart)

  manager/
    camera_manager.py
      # Thread-safe registry of CameraStream instances.
      # add/remove/list/stop_all cameras.

    config_store.py
      # Local cache of camera assignments + zone configs.
      # Written to disk on change. Read on startup.

  models/
    best_int8_openvino_model/
      # YOLOv8 INT8 OpenVINO model (.xml + .bin)
```

### 3.3 Per-Camera Processing Pipeline

```mermaid
graph TD
    RTSP["RTSP stream<br/>(1080p, 25-30 FPS native)"]

    VD["<b>VideoDecoder (iGPU)</b><br/>Decode priority:<br/>1. GStreamer + VA-API vaapidecodebin<br/>2. GStreamer + VA-API vah264dec<br/>3. GStreamer + QSV msdkh264dec<br/>4. FFmpeg + VA-API<br/>5. FFmpeg + QSV<br/>6. CPU software decode (last resort)<br/><br/>Output: raw frame to thread-safe buffer<br/>Rate: native FPS, latest frame only"]

    NPU["<b>NPUInferenceEngine (NPU)</b><br/>Shared across all cameras on this device.<br/>8 worker threads. Shared frame queue.<br/><br/>Per frame:<br/>1. Letterbox resize to 640x640<br/>2. Normalize 0,1<br/>3. INT8 inference on NPU<br/>4. NMS post-processing<br/>5. Output: list of detections<br/><br/>Performance: &lt;5ms per inference<br/>Aggregate: ~200 FPS (40 cams x 5 FPS)"]

    BBox["<b>Per-Camera BBox Filtering (CPU)</b><br/>conf_thr (default 0.25)<br/>min/max_bbox_area<br/>min/max_bbox_hw_ratio<br/><br/>Rejects bags, reflections, partial bodies"]

    BT["<b>ByteTrack Tracker (CPU)</b><br/>One instance per camera.<br/>Input: filtered detections<br/>Output: tracked detections with stable track_id<br/><br/>Config: track_thresh=0.25,<br/>track_buffer=30 (6s at 5 FPS),<br/>match_thresh=0.8<br/><br/>Latency: ~1-2ms per frame at 20 dets"]

    ZA1["Zone: Lane 1<br/>QueueAnalytics"]
    ZA2["Zone: Lane 2<br/>QueueAnalytics"]
    ZAN["Zone: ...<br/>QueueAnalytics"]

    MQTT["<b>MQTT Publisher (CPU + network)</b><br/>Per zone: metrics at 1 Hz<br/>Per zone: checkpoint at 1/30s<br/>Per zone: alerts on-event<br/>Per device: heartbeat at 1/5s<br/><br/>Offline buffer: SQLite WAL, 100 MB cap"]

    RTSP --> VD
    VD -->|"sampled at 5 FPS<br/>by CameraStream submit loop"| NPU
    NPU --> BBox
    BBox --> BT
    BT --> ZA1
    BT --> ZA2
    BT --> ZAN
    ZA1 --> MQTT
    ZA2 --> MQTT
    ZAN --> MQTT
```

### 3.4 MQTT Topics Published by Edge

| Topic | QoS | Frequency | Size | Content |
|---|---|---|---|---|
| `airport/edge/{edge_id}/zone/{zone_id}` | 1 | 1 Hz per zone | ~500 B | Zone metrics snapshot |
| `airport/edge/{edge_id}/heartbeat` | 1 (retained) | Every 5s | ~300 B | Device health |
| `airport/edge/{edge_id}/checkpoint/{zone_id}` | 1 | Every 30s per zone | ~1-3 KB | Active track state for failover |
| `airport/edge/{edge_id}/alert/{zone_id}` | 2 | On-event | ~200 B | Alert (queue/wait threshold) |

**MQTT Broker Security:** The Mosquitto broker uses per-device credentials and topic ACLs. Each edge device can only publish to its own `airport/edge/{edge_id}/` topics and subscribe to its own `airport/control/{edge_id}/` topics plus the broadcast topic. The cloud application server has admin access to all topics for monitoring and orchestration.

### 3.5 MQTT Topics Subscribed by Edge

| Topic | Purpose |
|---|---|
| `airport/control/{edge_id}/assign` | Camera add/remove commands with zone definitions |
| `airport/control/{edge_id}/config` | Zone polygon, threshold, confidence updates |
| `airport/control/broadcast` | Fleet-wide: stop_all, restart, model_update |

Edge devices authenticate to the MQTT broker with per-device credentials. Topic ACLs enforce that an edge can only publish to `airport/edge/{edge_id}/` where `{edge_id}` matches its own identity. This prevents a compromised edge from injecting data for other edges.

### 3.6 Message Schemas

**Zone Metrics (1 Hz per zone)**

```json
{
  "v": 1,
  "edge_id": "edge-01",
  "zone_id": "term-a-security-lane-3",
  "camera_id": "cam-a1-02",
  "ts": 1711792800.123,
  "queue_length": 12,
  "service_count": 3,
  "avg_waiting_time_s": 185.2,
  "avg_processing_time_s": 42.7,
  "predicted_wait_s": 210.5,
  "pred_wait_method": "predictive",
  "wait_p50_s": 165.0,
  "wait_p90_s": 290.0,
  "throughput_per_hour": 42.3,
  "active_track_count": 15,
  "inf_fps": 4.8,
  "inf_ms": 12.3
}
```

`zone_id` is the primary key for storage and display. `camera_id` and `edge_id` are metadata for infrastructure debugging.

**Edge Heartbeat (every 5s, retained)**

```json
{
  "v": 1,
  "edge_id": "edge-01",
  "ts": 1711792800.0,
  "uptime_s": 86400,
  "cameras_active": 28,
  "cameras_assigned": 30,
  "cameras_errored": 2,
  "zones_active": 55,
  "cpu_pct": 62.5,
  "mem_pct": 71.2,
  "npu_batches": 145023,
  "npu_avg_inf_ms": 4.2,
  "npu_dropped": 12,
  "error_cameras": ["cam-b4-01", "cam-b4-02"]
}
```

**Track Checkpoint (every 30s per zone)**

```json
{
  "v": 1,
  "edge_id": "edge-01",
  "zone_id": "term-a-security-lane-3",
  "camera_id": "cam-a1-02",
  "ts": 1711792800.0,
  "active_tracks": [
    {"track_id": 1042, "zone": "queue", "entry_ts": 1711792650.0},
    {"track_id": 1043, "zone": "service", "entry_ts": 1711792700.0}
  ],
  "recent_queue_dwells": [125.3, 142.1, 98.7, 110.5],
  "recent_service_dwells": [38.2, 45.1, 41.0, 39.8]
}
```

**Alert (on-event, QoS 2)**

```json
{
  "v": 1,
  "edge_id": "edge-01",
  "zone_id": "term-a-security-lane-3",
  "camera_id": "cam-a1-02",
  "ts": 1711792800.0,
  "alert_type": "queue_length",
  "level": "critical",
  "message": "Queue length 18 exceeds threshold 10",
  "queue_length": 18,
  "avg_wait_s": 245.0
}
```

**Camera Assignment Command (cloud -> edge)**

```json
{
  "action": "add",
  "cameras": [
    {
      "camera_id": "cam-a1-02",
      "source_url": "rtsp://10.0.1.102:554/stream1",
      "inf_fps": 5,
      "conf_threshold": 0.25,
      "zones": [
        {
          "zone_id": "term-a-security-lane-3",
          "name": "Security Lane 3",
          "queue_poly": [[0.0,0.0],[0.45,0.0],[0.45,1.0],[0.0,1.0]],
          "service_poly": [[0.45,0.0],[0.95,0.0],[0.95,1.0],[0.45,1.0]],
          "alert_threshold": 10,
          "wait_threshold": 300.0,
          "seed_state": {
            "active_tracks": [
              {"zone": "queue", "entry_ts": 1711792650.0},
              {"zone": "service", "entry_ts": 1711792700.0}
            ],
            "recent_queue_dwells": [125.3, 142.1, 98.7],
            "recent_service_dwells": [38.2, 45.1, 41.0]
          }
        },
        {
          "zone_id": "term-a-security-lane-4",
          "name": "Security Lane 4",
          "queue_poly": [[...]], "service_poly": [[...]],
          "alert_threshold": 10, "wait_threshold": 300.0
        }
      ]
    }
  ]
}
```

The `seed_state` is included only during failover (populated from the `track_checkpoints` table). For initial assignment or manual additions, `seed_state` is omitted and the zone starts fresh.

### 3.7 Edge Offline Behavior

When the cloud (MQTT broker) is unreachable:

1. **Processing continues uninterrupted.** The edge pipeline (decode -> inference -> tracking -> analytics) has zero cloud dependency.
2. **MQTT client buffers outbound messages** to a local SQLite WAL-mode database. Cap: 100 MB. At ~500 bytes/msg and ~60 zones/edge at 1 Hz (~30 KB/s per edge), this covers approximately 55 minutes of outage before oldest metrics are dropped. Alerts (QoS 2) are always retained.
3. **On reconnection**, the buffer drains. The cloud ingestion handles duplicates via `(zone_id, ts)` idempotent upserts.
4. **Configuration**: If the edge has a local config cache (`config_store.py`), it can restart and resume processing even without cloud connectivity.

### 3.8 Bandwidth Per Edge Device

| Message | Zones on this edge | Frequency | Bandwidth |
|---|---|---|---|
| Zone metrics | ~60 zones | 1 Hz each | ~30 KB/s |
| Checkpoints | ~60 zones | 1/30s each | ~4 KB/s |
| Heartbeat | 1 | 1/5s | ~0.1 KB/s |
| Alerts | ~60 zones | ~0.01/s each | negligible |
| **Per-edge total** | | | **~34 KB/s** |
| **Fleet total (15 edges)** | | | **~510 KB/s** |

---

## 4. Cloud Application Server

### 4.1 Architecture Decision: Modular Monolith

At 15-20 edge clients and ~800 zone metrics/second, microservices add operational complexity with no benefit. A single FastAPI process with internal module boundaries handles the load. If scaling is ever needed, split the MQTT consumer into a separate process first.

### 4.2 Tech Stack

| Component | Technology | Why |
|---|---|---|
| Web framework | FastAPI + uvicorn (4 async workers) | Async I/O handles 800 MQTT msgs/s + WebSocket connections + REST API. Same language as edge code. |
| MQTT broker | Mosquitto 2.x | 15-20 edge clients. Lightweight, stable, sufficient. EMQX if clustering is needed later. |
| Time-series DB | TimescaleDB (PostgreSQL 16 + extension) | SQL (familiar tooling), hypertables (auto time-partition), compression (10x on old data), continuous aggregates (pre-computed rollups). 800 inserts/s is trivial. |
| Cache + pub/sub | Redis 7.x | Latest zone metrics (hash per zone_id). Pub/sub for WebSocket gateway. |
| Dashboard transport | WebSocket (Starlette, built into FastAPI) | Server-push at 1 Hz. Selective subscriptions per browser client. |
| Reverse proxy | Caddy 2.x | Auto-TLS, WebSocket upgrade, static file serving for React SPA. |
| Frontend | React SPA | Virtual scrolling for 800 zones, area grouping, drill-down views. |

**API Key Authentication:** All mutation endpoints (POST, PUT, DELETE) are protected by an API key authentication middleware. The API key is configured via the `AQM_API_KEY` environment variable and must be passed in the `X-API-Key` request header. Read-only GET endpoints are accessible without authentication for dashboard consumption. Role-based access control (RBAC) with per-user permissions is planned for a future release.

### 4.3 Cloud Module Structure

```
cloud/
  main.py                          # FastAPI app entry. Mounts all routers.
  config.py                        # Environment-based config (DB URL, Redis URL, MQTT host).

  modules/
    ingestion/
      mqtt_consumer.py             # Connects to Mosquitto. Subscribes to:
                                   #   airport/edge/+/zone/+
                                   #   airport/edge/+/heartbeat
                                   #   airport/edge/+/checkpoint/+
                                   #   airport/edge/+/alert/+
                                   #
                                   # On each zone metrics message:
                                   #   1. Validate schema
                                   #   2. redis.hset("zone:{zone_id}:latest", data)
                                   #   3. redis.publish("metrics:{zone_id}", payload)
                                   #   4. Append to write batch (flushed to TimescaleDB every 1s)
                                   #
                                   # On heartbeat:
                                   #   redis.hset("edge:{edge_id}:heartbeat", data)
                                   #   Notify health_monitor
                                   #
                                   # On checkpoint:
                                   #   Upsert track_checkpoints table
                                   #
                                   # On alert:
                                   #   Insert alerts table
                                   #   redis.publish("alert:{zone_id}", payload)

    api/
      zones.py                     # GET /api/zones -- list all zones with latest metrics
                                   # GET /api/zones/{zone_id}/metrics -- latest (from Redis)
                                   # GET /api/zones/{zone_id}/history -- time-series (TimescaleDB)
                                   #     ?from=...&to=...&resolution=1m|5m|1h
                                   # POST /api/zones/{zone_id}/config -- update polygon/thresholds
                                   #     writes to DB + pushes to edge via MQTT
                                   #
                                   # POST /api/zones/batch -- batch zone import
                                   #     Accepts JSON array of zone definitions.
                                   #     Creates/updates multiple zones in a single transaction.
                                   #     Pushes config to all affected edges via MQTT.
                                   #
                                   # POST /api/zones/import -- CSV upload for bulk zone config
                                   #     Accepts multipart/form-data with CSV file.
                                   #     Columns: zone_id, name, area_id, camera_id,
                                   #              queue_poly, service_poly, alert_threshold,
                                   #              wait_threshold
                                   #     Returns summary: created, updated, errors.
                                   #
                                   # Zone templates:
                                   # POST /api/zones/templates -- create template
                                   #     Define a reusable shape + thresholds template.
                                   # GET /api/zones/templates -- list templates
                                   # POST /api/zones/templates/{template_id}/apply -- apply template
                                   #     Apply template to multiple cameras, creating zones
                                   #     with the template's polygon shape and thresholds.

      areas.py                     # GET /api/areas -- list areas
                                   # GET /api/areas/{area_id}/aggregate -- aggregated metrics
                                   #     (sum queue_length, avg wait, total throughput across zones)

      overview.py                  # GET /api/overview -- airport-wide summary
                                   #     total_queue, total_throughput, busiest_area,
                                   #     active_alert_count, edge_fleet_status

      alerts.py                    # GET /api/alerts -- list alerts (filterable by zone, area, level)
                                   # POST /api/alerts/{id}/ack -- acknowledge alert
                                   # GET /api/zones/{zone_id}/alerts -- alerts for specific zone

      edges.py                     # GET /api/edges -- fleet health (admin)
                                   # GET /api/edges/{edge_id} -- single edge + assigned cameras

      cameras.py                   # GET /api/cameras -- camera list (admin/infrastructure)
                                   # POST /api/cameras -- add new camera
                                   # DELETE /api/cameras/{camera_id} -- remove camera
                                   # POST /api/cameras/{camera_id}/reassign -- move to different edge

      predictions.py               # GET /api/zones/{zone_id}/predictions -- per-zone forecast
                                   #     ?horizon=30m|4h|7d|30d
                                   # GET /api/areas/{area_id}/predictions -- area aggregate forecast
                                   # GET /api/overview/predictions -- airport-wide forecast
                                   # GET /api/predictions/accuracy -- prediction quality metrics
                                   # POST /api/special-days -- add holiday/event to calendar
                                   # GET /api/special-days -- list special days
                                   # DELETE /api/special-days/{date}/{category} -- remove

      custom_metrics.py            # GET /api/custom-metrics -- list defined metrics
                                   # POST /api/custom-metrics -- create new metric definition
                                   # GET /api/custom-metrics/{id}/current -- latest value
                                   # GET /api/custom-metrics/{id}/history -- time-series
                                   # DELETE /api/custom-metrics/{id} -- remove metric

    orchestrator/
      health_monitor.py            # Runs on a background async task.
                                   # Checks edge heartbeat timestamps in Redis.
                                   # Escalation:
                                   #   No heartbeat 15s → status "warning", log
                                   #   No heartbeat 30s → status "offline", trigger failover
                                   #   Heartbeat resumes → status "recovering"

      camera_assigner.py           # Manages camera↔edge assignments.
                                   # Source of truth: cameras.assigned_edge in Postgres.
                                   # Uses Postgres advisory locks for atomic reassignment.
                                   #
                                   # initial_assign():
                                   #   Reads seed config YAML. Assigns by affinity.
                                   #   Pushes MQTT assign commands.
                                   #
                                   # failover(failed_edge_id):
                                   #   1. List cameras on failed edge
                                   #   2. List healthy edges sorted by capacity
                                   #   3. Round-robin distribute cameras
                                   #   4. Include last checkpoint seed_state in assign command
                                   #   5. Update DB, write assignment_log
                                   #
                                   # rebalance():
                                   #   Triggered when utilization imbalance > 30%
                                   #   Moves least-active cameras from busiest to quietest edge

    websocket/
      gateway.py                   # WS /ws/dashboard
                                   # On connect: client sends subscription filter
                                   # Subscribes to matching Redis pub/sub channels
                                   # Forwards messages to browser at up to 1 Hz per zone
                                   #
                                   # Also computes area aggregates in-memory:
                                   #   Maintains latest metric per zone in a dict.
                                   #   On any zone update, reaggregates its area.
                                   #   Pushes area-level update to area subscribers.

      subscriptions.py             # Per-client subscription management.
                                   # Supports patterns:
                                   #   "overview" → airport-wide aggregate (1 msg/s)
                                   #   "area:term-a-security" → all zones in area (~20 msg/s)
                                   #   "zone:term-a-sec-lane-3" → single zone (1 msg/s)

    predictions/
      forecast_engine.py           # Generates predictions at multiple horizons.
                                   # Runs on a schedule (not real-time):
                                   #   Every 5 min: regenerate 30-60 min predictions
                                   #   Every 1 hour: regenerate 2-4 hr and 7-day predictions
                                   #   Every 6 hours: regenerate 30-day predictions
                                   #
                                   # Reads from zone_metrics_1m and zone_metrics_1h aggregates.
                                   # Writes to zone_predictions table.
                                   # Publishes updated predictions to Redis for dashboard.

      pattern_model.py             # Historical pattern matching.
                                   # For a given (zone_id, future_timestamp):
                                   #   1. Find same day-of-week, same time-of-day in history
                                   #   2. Weight by recency (last 4 weeks > older)
                                   #   3. Separate models for normal days vs special days
                                   #   4. Adjust by "today factor": how today is tracking
                                   #      vs historical average at current time
                                   #
                                   # Output: predicted queue_length, wait_s, throughput
                                   #         with confidence band (low, expected, high)

      special_days.py              # Holiday/special day calendar management.
                                   # Categories: public_holiday, school_holiday,
                                   #   major_event, schedule_change, reduced_capacity
                                   # Each category learns its own historical pattern.

      accuracy_tracker.py          # Compares past predictions against actuals.
                                   # Logs MAPE (mean absolute percentage error) per zone per horizon.
                                   # Stored in zone_prediction_accuracy table.
                                   # Exposed on dashboard so operators see prediction quality.

      flight_adapter.py            # Phase 2: plugs flight schedule data into predictions.
                                   # Accepts arrival/departure schedule + pax counts.
                                   # Adjusts baseline predictions from pattern_model.
                                   # Not active until flight data feed is configured.

    custom_metrics/
      metric_definitions.py        # User-defined composite metrics.
                                   # Operators define metrics like:
                                   #   "Total Pax Arrived" = sum of service_exits across
                                   #     [zone-a, zone-b, zone-c] (entry zones)
                                   #   "Terminal A Total Queuing" = sum of queue_length
                                   #     across all zones in area "term-a-*"
                                   #   "Security Throughput Rate" = sum of throughput
                                   #     across zones tagged "security"
                                   #
                                   # Definitions stored in custom_metrics table.
                                   # Evaluated every second from Redis latest state.
                                   # Historical values computed from zone_metrics aggregates.
                                   # Available on dashboard like any other metric.

  models/
    db.py                          # SQLAlchemy async models / asyncpg connection pool
    schemas.py                     # Pydantic models for API requests/responses + MQTT messages
```

### 4.4 Ingestion Data Flow (Detail)

```mermaid
graph TD
    MQTT["MQTT Broker<br/>(Mosquitto)<br/>airport/edge/+/zone/+"]

    Consumer["<b>mqtt_consumer.py</b><br/>1. JSON parse + Pydantic validate<br/>2. Dedup check: skip if (zone_id,ts)<br/>already in current write batch"]

    RedisHSET["Redis HSET<br/>latest state<br/>zone:X:latest (hash)<br/>TTL: 30s"]
    RedisPUB["Redis PUBLISH<br/>pub/sub channel"]
    WriteBatch["Write Batch"]

    WSGateway["WebSocket Gateway<br/>Filters by client sub patterns<br/>Pushes to browser WS"]
    TimescaleDB["TimescaleDB<br/>zone_metrics hypertable<br/>Batch INSERT ~800 rows/s"]

    Browser["Browser"]

    MQTT --> Consumer
    Consumer --> RedisHSET
    Consumer --> RedisPUB
    Consumer --> WriteBatch
    RedisPUB --> WSGateway
    WriteBatch -->|"flush every 1s"| TimescaleDB
    WSGateway --> Browser
```

### 4.5 How Zone Configuration Changes Propagate

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant Postgres
    participant MQTTBroker as MQTT Broker
    participant Edge

    Browser->>FastAPI: POST /api/zones/{zone_id}/config<br/>{queue_poly, service_poly, alert_threshold, wait_threshold}
    FastAPI->>FastAPI: Validate polygon geometry
    FastAPI->>Postgres: UPDATE zones table
    FastAPI->>Postgres: Look up camera → edge mapping
    Postgres-->>FastAPI: edge_id, camera_id
    FastAPI->>MQTTBroker: PUBLISH airport/control/{edge_id}/config<br/>{camera_id, zone_id, queue_poly, service_poly, thresholds}
    FastAPI-->>Browser: 200 OK
    MQTTBroker->>Edge: Deliver config message
    Edge->>Edge: Find CameraStream for camera_id
    Edge->>Edge: Find QueueAnalytics for zone_id
    Edge->>Edge: Call analytics.set_zones(q, s, e)<br/>(resets debounce, clears stale dwell tracking)
    Edge->>Edge: Update local config cache
    Edge->>Edge: Analytics resumes with new polygons
```

---

## 5. Data Flow

### 5.1 End-to-End: Camera Frame -> Dashboard Pixel

```mermaid
graph TD
    subgraph EDGE["EDGE DEVICE"]
        RTSP["RTSP Camera"]
        iGPU["iGPU decode"]
        FrameBuf["frame buffer"]
        Sample["5 FPS sample"]
        NPU["NPU inference<br/>(YOLOv8 INT8)"]
        BBoxFilter["BBox size/ratio filter"]
        ByteTrack["ByteTrack<br/>(track IDs)"]

        ZoneLane3["Zone 'Lane 3'<br/>QueueAnalytics"]
        ZoneLane4["Zone 'Lane 4'<br/>QueueAnalytics"]
        ZoneN["Zone '...'<br/>QueueAnalytics"]

        MQTTPub["MQTT publish<br/>(1 Hz per zone)"]

        RTSP --> iGPU --> FrameBuf --> Sample --> NPU
        NPU --> BBoxFilter --> ByteTrack
        ByteTrack --> ZoneLane3
        ByteTrack --> ZoneLane4
        ByteTrack --> ZoneN
        ZoneLane3 --> MQTTPub
        ZoneLane4 --> MQTTPub
        ZoneN --> MQTTPub
    end

    LAN["Airport LAN/WAN"]
    MQTTPub --> LAN

    subgraph CLOUD["CLOUD"]
        Broker["MQTT Broker<br/>(Mosquitto)"]
        MQTTCons["mqtt_consumer.py"]
        RedisHSET2["Redis HSET<br/>(latest)"]
        RedisPUB2["Redis PUBLISH<br/>(pub/sub)"]
        TSDB["TimescaleDB<br/>batch INSERT<br/>(history)"]
        WSGateway2["WebSocket Gateway"]
        SubFilter["subscription filter<br/>(per browser client)"]

        Broker --> MQTTCons
        MQTTCons --> RedisHSET2
        MQTTCons --> RedisPUB2
        MQTTCons --> TSDB
        RedisPUB2 --> WSGateway2
        WSGateway2 --> SubFilter
    end

    LAN --> Broker

    BrowserApp["Browser<br/>React SPA<br/><br/>REST API: initial load<br/>WebSocket: real-time updates"]

    RedisHSET2 --> BrowserApp
    SubFilter --> BrowserApp
```

### 5.2 Latency Budget (Frame -> Dashboard)

| Step | Typical Latency |
|---|---|
| RTSP decode | ~5ms |
| Frame queue wait | 0-200ms (5 FPS -> 200ms cadence) |
| NPU inference | 3-5ms |
| BBox filter + ByteTrack | 1-2ms |
| Zone analytics (per zone) | 0.5ms |
| MQTT publish (edge -> broker) | 1-5ms (LAN) |
| MQTT -> consumer -> Redis | 1-2ms |
| Redis pub/sub -> WebSocket | <1ms |
| WebSocket -> browser render | 5-10ms |
| **Total** | **~220-430ms** |

Operators see zone metrics with <0.5s latency from the physical world.

---

## 6. Database Schema

### 6.1 Full Schema (TimescaleDB / PostgreSQL 16)

```sql
-- ===================================================================
-- CONFIGURATION TABLES (regular PostgreSQL)
-- ===================================================================

CREATE TABLE areas (
    area_id         TEXT PRIMARY KEY,                -- "term-a-security"
    name            TEXT NOT NULL,                   -- "Terminal A Security"
    terminal        TEXT,                            -- "A"
    category        TEXT,                            -- "security" | "check-in" | "immigration" | "gate"
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE edges (
    edge_id         TEXT PRIMARY KEY,                -- "edge-01"
    hostname        TEXT,                            -- "edge-01.airport.local"
    ip_address      INET,
    max_cameras     INT DEFAULT 35,
    last_heartbeat  TIMESTAMPTZ,
    status          TEXT DEFAULT 'unknown',          -- "online" | "warning" | "offline" | "recovering"
    meta            JSONB DEFAULT '{}'               -- latest heartbeat payload (cpu_pct, mem_pct, etc.)
);

CREATE TABLE cameras (
    camera_id       TEXT PRIMARY KEY,                -- "cam-a1-02"
    name            TEXT,                            -- "CAM-A1-02" (physical label)
    source_url      TEXT NOT NULL,                   -- "rtsp://10.0.1.102:554/stream1"
    assigned_edge   TEXT REFERENCES edges(edge_id),
    status          TEXT DEFAULT 'unknown',          -- "online" | "offline" | "stream_error"
    conf_threshold  REAL DEFAULT 0.25,
    inf_fps         REAL DEFAULT 5.0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE zones (
    zone_id         TEXT PRIMARY KEY,                -- "term-a-security-lane-3"
    name            TEXT NOT NULL,                   -- "Security Lane 3"
    area_id         TEXT REFERENCES areas(area_id),
    camera_id       TEXT REFERENCES cameras(camera_id),
    queue_poly      JSONB,                           -- [[0,0],[0.5,0],[0.5,1],[0,1]]  (normalized)
    service_poly    JSONB,                           -- [[0.5,0],[1,0],[1,1],[0.5,1]]
    exit_poly       JSONB,                           -- optional
    alert_threshold INT DEFAULT 10,
    wait_threshold  REAL DEFAULT 300.0,              -- seconds
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ===================================================================
-- TIME-SERIES DATA (TimescaleDB hypertables)
-- ===================================================================

CREATE TABLE zone_metrics (
    ts                      TIMESTAMPTZ NOT NULL,
    zone_id                 TEXT NOT NULL,
    camera_id               TEXT,                    -- which camera fed this (metadata)
    edge_id                 TEXT,                    -- which edge computed this (metadata)
    queue_length            SMALLINT,
    service_count           SMALLINT,
    avg_waiting_time_s      REAL,
    avg_processing_time_s   REAL,
    predicted_wait_s        REAL,
    pred_wait_method        TEXT,                    -- "predictive" | "fallback" | "warming_up"
    wait_p50_s              REAL,
    wait_p90_s              REAL,
    throughput_per_hour     REAL,
    active_track_count      SMALLINT,
    inf_fps                 REAL,
    inf_ms                  REAL
);

SELECT create_hypertable('zone_metrics', 'ts');
CREATE INDEX idx_zm_zone_ts ON zone_metrics (zone_id, ts DESC);

-- Compression: segmented by zone_id for efficient per-zone queries
ALTER TABLE zone_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'zone_id',
    timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('zone_metrics', INTERVAL '7 days');

-- 1-minute continuous aggregate
CREATE MATERIALIZED VIEW zone_metrics_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts) AS bucket,
    zone_id,
    AVG(queue_length)::REAL            AS avg_queue,
    MAX(queue_length)                  AS max_queue,
    AVG(avg_waiting_time_s)::REAL      AS avg_wait_s,
    MAX(avg_waiting_time_s)::REAL      AS max_wait_s,
    AVG(predicted_wait_s)::REAL        AS avg_predicted_wait_s,
    AVG(wait_p90_s)::REAL              AS avg_wait_p90_s,
    AVG(throughput_per_hour)::REAL     AS avg_throughput
FROM zone_metrics
GROUP BY bucket, zone_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('zone_metrics_1m',
    start_offset  => INTERVAL '10 minutes',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute'
);

-- 1-hour continuous aggregate (built on 1-minute)
CREATE MATERIALIZED VIEW zone_metrics_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', bucket)  AS bucket,
    zone_id,
    AVG(avg_queue)::REAL           AS avg_queue,
    MAX(max_queue)                 AS max_queue,
    AVG(avg_wait_s)::REAL          AS avg_wait_s,
    MAX(max_wait_s)::REAL          AS max_wait_s,
    AVG(avg_throughput)::REAL      AS avg_throughput
FROM zone_metrics_1m
GROUP BY bucket, zone_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('zone_metrics_1h',
    start_offset  => INTERVAL '3 hours',
    end_offset    => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- Retention: raw 90 days, 1-min rollups 1 year, 1-hour rollups 2 years
SELECT add_retention_policy('zone_metrics', INTERVAL '90 days');
SELECT add_retention_policy('zone_metrics_1m', INTERVAL '365 days');

-- ===================================================================
-- FAILOVER STATE
-- ===================================================================

CREATE TABLE track_checkpoints (
    zone_id         TEXT PRIMARY KEY,                -- upserted on each checkpoint
    edge_id         TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    active_tracks   JSONB NOT NULL,                  -- [{zone:"queue", entry_ts:...}, ...]
    dwell_history   JSONB NOT NULL                   -- {queue: [125.3, ...], service: [38.2, ...]}
);

-- ===================================================================
-- ALERTS
-- ===================================================================

CREATE TABLE alerts (
    alert_id        BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL,
    zone_id         TEXT NOT NULL,
    camera_id       TEXT,
    edge_id         TEXT,
    alert_type      TEXT NOT NULL,                   -- "queue_length" | "wait_time" | "open_counter"
    level           TEXT NOT NULL,                   -- "warning" | "critical"
    message         TEXT,
    queue_length    SMALLINT,
    avg_wait_s      REAL,
    acknowledged    BOOLEAN DEFAULT FALSE,
    acked_by        TEXT,
    acked_at        TIMESTAMPTZ
);

CREATE INDEX idx_alerts_zone ON alerts (zone_id, ts DESC);
CREATE INDEX idx_alerts_unacked ON alerts (acknowledged, ts DESC) WHERE NOT acknowledged;

-- ===================================================================
-- PREDICTIONS
-- ===================================================================

-- Prediction results (regenerated periodically by forecast_engine)
CREATE TABLE zone_predictions (
    zone_id             TEXT NOT NULL,
    generated_at        TIMESTAMPTZ NOT NULL,        -- when the prediction was computed
    target_ts           TIMESTAMPTZ NOT NULL,        -- the future timestamp being predicted
    horizon             TEXT NOT NULL,                -- "30m" | "4h" | "7d" | "30d"
    predicted_queue     REAL,
    predicted_queue_lo  REAL,                         -- confidence band low
    predicted_queue_hi  REAL,                         -- confidence band high
    predicted_wait_s    REAL,
    predicted_wait_lo   REAL,
    predicted_wait_hi   REAL,
    predicted_throughput REAL,
    PRIMARY KEY (zone_id, target_ts, horizon)
);
CREATE INDEX idx_zp_zone_target ON zone_predictions (zone_id, target_ts);

-- Special day calendar
CREATE TABLE special_days (
    date                DATE NOT NULL,
    category            TEXT NOT NULL,                -- "public_holiday" | "school_holiday" | "major_event" | etc.
    name                TEXT,                         -- "Easter Monday", "World Cup Final"
    PRIMARY KEY (date, category)
);

-- Prediction accuracy tracking
CREATE TABLE prediction_accuracy (
    zone_id             TEXT NOT NULL,
    target_ts           TIMESTAMPTZ NOT NULL,
    horizon             TEXT NOT NULL,
    predicted_queue     REAL,
    actual_queue        REAL,
    mape_pct            REAL,                         -- mean absolute percentage error
    evaluated_at        TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (zone_id, target_ts, horizon)
);

-- ===================================================================
-- CUSTOM METRICS
-- ===================================================================

-- User-defined composite metrics (e.g., "Total Pax Arrived", "Security Throughput")
CREATE TABLE custom_metrics (
    metric_id           TEXT PRIMARY KEY,             -- "total-pax-arrived"
    name                TEXT NOT NULL,                -- "Total Pax Arrived"
    description         TEXT,
    aggregation         TEXT NOT NULL,                -- "sum" | "avg" | "max" | "min"
    source_field        TEXT NOT NULL,                -- "queue_length" | "service_exits" | "throughput_per_hour" etc.
    zone_filter         TEXT NOT NULL,                -- zone selection: area_id, zone tag, or explicit zone_id list
    zone_filter_type    TEXT NOT NULL,                -- "area" | "tag" | "zone_list"
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Historical values for custom metrics (computed from zone_metrics aggregates)
CREATE TABLE custom_metric_values (
    ts                  TIMESTAMPTZ NOT NULL,
    metric_id           TEXT NOT NULL REFERENCES custom_metrics(metric_id),
    value               REAL
);
SELECT create_hypertable('custom_metric_values', 'ts');
SELECT add_compression_policy('custom_metric_values', INTERVAL '7 days');
SELECT add_retention_policy('custom_metric_values', INTERVAL '90 days');

-- ===================================================================
-- AUDIT LOG
-- ===================================================================

CREATE TABLE assignment_log (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ DEFAULT NOW(),
    camera_id       TEXT NOT NULL,
    from_edge       TEXT,
    to_edge         TEXT,
    reason          TEXT                             -- "initial" | "failover" | "rebalance" | "manual"
);
```

### 6.2 Storage Estimates

| Data | Daily Volume | With Compression | Retention | Total Disk |
|---|---|---|---|---|
| Raw zone_metrics | ~34 GB/day (800 rows/s x 500 B) | ~3.4 GB/day (10x) | 90 days | ~306 GB |
| zone_metrics_1m | ~55 MB/day | -- | 1 year | ~20 GB |
| zone_metrics_1h | ~2 MB/day | -- | 2 years | ~1.5 GB |
| alerts | negligible | -- | indefinite | <1 GB |
| **Total** | | | | **~330 GB** |

### 6.3 Query Resolution Selection

The API automatically selects the right data source based on the requested time range:

| Requested Range | Source | Resolution | Why |
|---|---|---|---|
| Last 10 minutes | `zone_metrics` (raw) | 1 second | Most recent, uncompressed, fast |
| Last 1-24 hours | `zone_metrics_1m` | 1 minute | Continuous aggregate, pre-computed |
| 1 day - 90 days | `zone_metrics_1m` or `_1h` | 1 min or 1 hour | Compressed raw or rollup |
| 90 days+ | `zone_metrics_1h` | 1 hour | Only source available |

---

## 7. Dashboard & WebSocket

### 7.1 Dashboard Hierarchy

```mermaid
graph TD
    Overview["<b>AIRPORT OVERVIEW</b><br/>Total queuing: 342 | Avg wait: 3m 12s<br/>Throughput: 1,240/hr | Alerts: 7"]

    AreaA["<b>Term A Security</b><br/>Queue: 85 | Avg wait: 4m<br/>Status: RED"]
    AreaB["<b>Term B Check-in</b><br/>Queue: 120 | Avg wait: 2m<br/>Status: YELLOW"]
    AreaC["<b>Gate C Boarding</b><br/>Queue: 45 | Avg wait: 8m<br/>Status: RED"]

    Overview -->|click| AreaA
    Overview --> AreaB
    Overview --> AreaC

    Lane1["<b>Lane 1</b><br/>Q: 18 | W: 5m | T: 42/h<br/>RED"]
    Lane2["<b>Lane 2</b><br/>Q: 12 | W: 3m | T: 48/h<br/>GREEN"]
    Lane3["<b>Lane 3</b><br/>Q: 22 | W: 7m | T: 35/h<br/>RED"]
    Lane4["<b>Lane 4</b><br/>Q: 15 | W: 4m | T: 40/h<br/>YELLOW"]
    Lane5["<b>Lane 5</b><br/>Q: 18 | W: 5m | T: 38/h<br/>RED"]

    AreaA -->|click| Lane1
    AreaA --> Lane2
    AreaA --> Lane3
    AreaA --> Lane4
    AreaA --> Lane5

    Detail["<b>ZONE DETAIL: Security Lane 1</b><br/>Queue: 18 | Wait: 5m 12s | Processing: 45s<br/>Predicted: 6m | P50: 4m 30s | P90: 8m 15s<br/>Throughput: 42/hr<br/><br/>Queue Length Chart (last 30 min)<br/>Wait Time Chart (last 30 min)<br/><br/>Alerts:<br/>12:34 CRITICAL Queue 18 > threshold 10<br/>12:28 WARNING Wait 5m > threshold 5m<br/><br/>Infrastructure: Camera CAM-A1-01 → Edge-01 | Status: NORMAL"]

    Lane1 -->|click| Detail
```

### 7.2 WebSocket Protocol

**Connection:**

```
Browser opens WS to wss://dashboard.airport.com/ws/dashboard
```

**Subscribe (browser -> server):**

```json
{"action": "subscribe", "topics": ["overview"]}
{"action": "subscribe", "topics": ["area:term-a-security"]}
{"action": "subscribe", "topics": ["zone:term-a-sec-lane-1"]}
{"action": "unsubscribe", "topics": ["overview"]}
```

**Zone metric update (server -> browser):**

```json
{
  "type": "zone_metric",
  "zone_id": "term-a-sec-lane-1",
  "data": {
    "queue_length": 18,
    "avg_waiting_time_s": 312.0,
    "predicted_wait_s": 360.0,
    "pred_wait_method": "predictive",
    "wait_p50_s": 270.0,
    "wait_p90_s": 495.0,
    "throughput_per_hour": 42.3,
    "service_count": 3,
    "status": "normal"
  }
}
```

**Area aggregate (server -> browser, computed in-memory by gateway):**

```json
{
  "type": "area_aggregate",
  "area_id": "term-a-security",
  "data": {
    "total_queue": 85,
    "avg_wait_s": 240.0,
    "total_throughput_per_hour": 203.0,
    "zone_count": 5,
    "alert_count": 3,
    "busiest_zone": "term-a-sec-lane-3"
  }
}
```

**Alert (server -> browser, pushed immediately):**

```json
{
  "type": "alert",
  "zone_id": "term-a-sec-lane-1",
  "data": {
    "alert_id": 4523,
    "alert_type": "queue_length",
    "level": "critical",
    "message": "Queue length 18 exceeds threshold 10",
    "ts": "2026-03-30T12:34:00Z"
  }
}
```

**Zone status change (server -> browser):**

```json
{
  "type": "zone_status",
  "zone_id": "term-a-sec-lane-1",
  "status": "recovering",
  "last_update_ts": "2026-03-30T12:33:45Z"
}
```

### 7.3 Selective Subscription Bandwidth

| View | Subscription | Messages/s to this client |
|---|---|---|
| Airport Overview | `overview` | 1 (pre-aggregated) |
| Area View (20 zones) | `area:term-a-security` | ~20 |
| Zone Detail | `zone:term-a-sec-lane-1` | 1 |
| All zones (admin) | `zone:*` | ~800 (not recommended) |

---

## 8. Failover & Resilience

### 8.1 Edge Failure Detection

```mermaid
stateDiagram-v2
    [*] --> ONLINE : HB recv'd

    ONLINE --> WARNING : no HB 15s<br/>(3 missed heartbeats)
    WARNING --> ONLINE : HB recv'd
    WARNING --> OFFLINE : no HB 30s<br/>(6 missed heartbeats)
    OFFLINE --> RECOVERING : HB recv'd
    RECOVERING --> ONLINE : 5 consecutive HBs

    note right of ONLINE
        Heartbeat interval: 5s
    end note

    note right of RECOVERING
        stop_all sent,
        may get cameras back
    end note
```

### 8.2 Camera Reassignment Algorithm

When the orchestrator marks an edge `OFFLINE`:

1. **ACQUIRE** Postgres advisory lock (prevents concurrent reassignment).

2. **SELECT** camera_id, source_url FROM cameras WHERE assigned_edge = '{failed_edge_id}'.

3. **SELECT** edge_id, max_cameras, cameras_active FROM edges WHERE status = 'online' ORDER BY (max_cameras - cameras_active) DESC (edges with most headroom first).

4. **For each orphaned camera:**
   - a. Pick next healthy edge (round-robin across those with capacity).
   - b. SELECT * FROM zones WHERE camera_id = this camera.
   - c. For each zone on this camera: SELECT * FROM track_checkpoints WHERE zone_id = this zone (get last checkpoint for seed_state).
   - d. Publish MQTT assign command to target edge (with zone configs + seed_state).
   - e. UPDATE cameras SET assigned_edge = target_edge.
   - f. INSERT assignment_log (camera, from, to, reason='failover').

5. **RELEASE** advisory lock.

**Example:** Edge-01 fails with 30 cameras. 14 surviving edges absorb ~2 cameras each. No single edge gets hammered.

### 8.3 Stats Continuity During Failover

**What's safe:** All completed dwell times and historical zone_metrics are in TimescaleDB. These survive any infrastructure change.

**What's lost:** Active tracks on the failed edge -people currently standing in a queue whose `entry_ts` was recorded in the failed edge's memory.

**Recovery approach: Checkpoint seeding + cloud-side smoothing**

```
Timeline of a failover event:

t=0s     Edge-01 fails (process crash, hardware failure, network cut)
         Dashboard: zones fed by Edge-01 cameras continue showing
                    last-known values (stale but reasonable)

t=10s    No metrics for 10s → dashboard marks those zones "STALE"
         (yellow badge, grayed values, "Last update: 10s ago")

t=15s    Orchestrator: Edge-01 heartbeat missing 15s → status "WARNING"

t=30s    Orchestrator: Edge-01 heartbeat missing 30s → status "OFFLINE"
         Orchestrator: triggers camera reassignment
         Dashboard: affected zones show "FAILOVER IN PROGRESS" (orange badge)

t=30-35s Orchestrator distributes 30 cameras across 14 surviving edges
         Each assign command includes:
          -RTSP URL, zone polygons, thresholds
          -seed_state from last checkpoint (≤30s old):
            -recent_queue_dwells → seeds p50/p90 immediately
            -active_tracks with entry_ts → seeds wait estimates
         New edges start CameraStream + analytics for assigned cameras

t=35-40s New edges decode first frames, NPU inference starts,
         ByteTrack begins building tracks from scratch

t=40-45s First metrics arrive from new edges for reassigned zones
         Dashboard: zones switch to "RECOVERING" (blue badge)

         Cloud applies smoothing during recovery window:
           alpha = 0 → 1 over 120 seconds (exponential ramp)
           displayed_wait = alpha × new_edge_value
                          + (1 - alpha) × last_known_good_value

         This prevents the dashboard from showing "0 wait" → "5 min wait"
         as the new edge's tracker stabilizes

t=90-120s ByteTrack has stable tracks. At least one queue→exit transition
          has occurred. pred_wait_method switches to "predictive".
          Dashboard: zones return to "NORMAL" (badge removed)

Total outage:     ~30s (metrics gap)
Degraded accuracy: ~60-90s (recovering phase)
Data loss:         Zero (historical) / ≤30s staleness (active tracks)
```

### 8.4 Zone Status State Machine

Each zone in the dashboard follows this state machine:

```mermaid
stateDiagram-v2
    [*] --> NORMAL

    NORMAL --> STALE : no metric for 10s
    STALE --> FAILOVER : orchestrator assigns\nto new edge
    FAILOVER --> RECOVERING : first metric from\nnew edge arrives
    RECOVERING --> NORMAL : pred_method == "predictive"\nfor 5 consecutive updates

    note right of STALE
        yellow badge,
        grayed values
    end note

    note right of FAILOVER
        orange badge
    end note

    note right of RECOVERING
        blue badge,
        smoothed values
    end note
```

### 8.5 Recovery & Split-Brain Prevention

When a previously-offline edge comes back online:

1. Orchestrator detects heartbeat from recovered edge.
2. **The recovered edge does NOT automatically reclaim cameras.** This prevents split-brain (two edges processing the same camera simultaneously).
3. Orchestrator sends `{"action": "stop_all"}` to the recovered edge.
4. The recovered edge stops any cameras it was still processing.
5. Orchestrator evaluates fleet load:
  -If surviving edges are near capacity -> rebalance some cameras back to the recovered edge.
  -If fleet is comfortable -> leave assignments as-is.
6. Any reassignment uses the same MQTT assign flow as failover.

**Source of truth**: `cameras.assigned_edge` column in PostgreSQL. All assignment changes go through Postgres advisory locks. If an edge receives inference results for a camera it's no longer assigned to (checked against its local assignment cache, refreshed from MQTT), it drops the results.

### 8.6 Camera Health Monitoring

```mermaid
stateDiagram-v2
    [*] --> INIT

    INIT --> ONLINE : decode success
    INIT --> STREAM_ERROR : decode fails\nafter 30s

    ONLINE --> STREAM_ERROR : decode fails\n(5 consecutive)

    STREAM_ERROR --> ONLINE : decode succeeds

    note right of STREAM_ERROR
        Edge retries every 10s.
        Dashboard shows "STREAM ERR"
        on affected zones.
        Alert raised for network/facilities team.

        Camera STREAM_ERROR does NOT
        trigger failover — another edge
        would have the same network problem.
    end note
```

---

## 9. Operations

### 9.1 Adding a New Camera

```
Operator: "We installed a new camera at Gate D2"

1. POST /api/cameras
   {
     "camera_id": "cam-d2-01",
     "name": "CAM-D2-01",
     "source_url": "rtsp://10.0.4.201:554/stream1"
   }

2. Cloud orchestrator:
   - Selects edge with most headroom
   - Publishes MQTT assign (camera only, no zones yet)
   - Edge starts decoding + inference (no analytics yet, no zones defined)

3. Operator opens Zone Setup in dashboard:
   - Selects camera cam-d2-01
   - Draws queue polygon on live snapshot
   - Draws service polygon
   - Names the zone: "Gate D2 Queue"
   - Sets thresholds

4. POST /api/zones
   {
     "zone_id": "gate-d2-queue",
     "name": "Gate D2 Queue",
     "area_id": "gate-d",
     "camera_id": "cam-d2-01",
     "queue_poly": [[...]],
     "service_poly": [[...]],
     "alert_threshold": 8,
     "wait_threshold": 240.0
   }

5. Cloud:
   -Writes zone to DB
   -Pushes config to edge via MQTT
   -Edge creates QueueAnalytics instance for this zone
   -Metrics start flowing within seconds

Zero downtime. No restarts.
```

### 9.2 Adding a New Zone to an Existing Camera

```
Operator: "Camera CAM-B2-01 can also see Counter C"

1. POST /api/zones
   {
     "zone_id": "term-b-checkin-counter-c",
     "name": "Counter C",
     "area_id": "term-b-checkin",
     "camera_id": "cam-b2-01",
     "queue_poly": [[...]],
     "service_poly": [[...]]
   }

2. Cloud pushes zone config to edge via MQTT
3. Edge adds another QueueAnalytics instance for this camera
4. Inference was already running on this camera -no extra NPU load
5. CPU overhead: +0.5ms per frame for the new zone (negligible)
6. Metrics for "Counter C" start appearing on dashboard
```

### 9.3 Removing a Zone

```
DELETE /api/zones/term-b-checkin-counter-c

1. Cloud marks zone inactive in DB
2. Pushes removal to edge via MQTT
3. Edge stops QueueAnalytics instance for that zone
4. Historical data remains in TimescaleDB (queryable, not displayed)
```

### 9.4 Removing a Camera

```
DELETE /api/cameras/cam-d2-01

1. Cloud checks: any active zones on this camera?
   If yes: returns 409 Conflict ("deactivate zones first")
2. Pushes MQTT remove to assigned edge
3. Edge stops VideoDecoder + inference for this camera
4. Camera removed from DB assignment
```

### 9.5 Upgrading the Detection Model

```
Scenario: retrained YOLOv8 with better accuracy, new INT8 model files

1. Build new Docker image with updated model files in edge_agent/models/
   or: use Ansible to copy model files to all edges

2. Rolling update via Ansible (one edge at a time):

   For each edge:
     a. Orchestrator: drain cameras from this edge
        (reassign to other edges via failover mechanism)
     b. Wait for drain confirmation (edge reports 0 cameras active)
     c. Stop edge container
     d. Pull new image / copy new model files
     e. Start edge container
     f. Edge announces heartbeat → orchestrator marks it "online"
     g. Orchestrator: reassign cameras back to this edge
     h. Health check: verify metrics flowing for all zones
     i. Proceed to next edge

Alternatively (hot reload, no drain needed):
   a. Copy new model dir to edge via Ansible
   b. Send MQTT broadcast: {"action": "reload_model"}
   c. Each edge's NPUInferenceEngine hot-reloads the new model
      (existing OpenVINO model candidate list already supports this)
   d. Brief inference pause (~2-3s while new model compiles on NPU)
   e. Resume with new model

Total fleet upgrade time:
  Rolling with drain: ~5 min per edge × 15 edges = ~75 min
  Hot reload: ~30s fleet-wide
```

### 9.6 Adding a New Edge Device

```
Scenario: airport expands, need a 16th edge device

1. Provision hardware (Intel Core Ultra 7 255H)
2. Install Ubuntu 22.04 + Intel NPU drivers + OpenVINO
3. Deploy edge agent Docker image via Ansible
4. Configure MQTT broker address + edge_id in environment
5. Start edge agent
6. Edge publishes heartbeat → cloud auto-discovers it
7. Orchestrator: adds edge to DB with status "online"
8. Assign cameras via:
   -Manual: POST /api/cameras/{cam}/reassign {"edge_id": "edge-16"}
   -Auto-rebalance: orchestrator detects imbalance, moves cameras

No restart of other edges or cloud server required.
```

### 9.7 Edge Fleet Management

| Task | Method | Downtime |
|---|---|---|
| Deploy new code | Ansible rolling update (drain -> update -> restore) | ~5 min per edge |
| Deploy new model (hot) | Ansible copy + MQTT reload_model broadcast | ~30s fleet-wide |
| Deploy new model (cold) | Ansible rolling update | ~5 min per edge |
| OS security patches | Ansible + reboot, one edge at a time with drain | ~10 min per edge |
| Replace failed hardware | Provision new device, Ansible deploy, orchestrator assigns cameras | ~30 min |
| Add edge device | Provision + deploy + orchestrator auto-discovers | ~30 min |
| Emergency stop all | MQTT broadcast: `{"action": "stop_all"}` | Immediate |

### 9.8 Monitoring Stack

```mermaid
graph TD
    subgraph EDGE["Each Edge Device"]
        NodeExp["prometheus node_exporter<br/>CPU, memory, disk, network"]
        IntelGPU["intel_gpu_top metrics<br/>iGPU utilization, NPU utilization"]
        Heartbeat["edge_agent heartbeat<br/>cameras active, inference FPS,<br/>NPU dropped frames, errors"]
        Promtail["promtail (log shipper)<br/>ships queue_system.log to Loki"]
    end

    Prometheus["Prometheus<br/>(scrapes all edges)"]
    Loki["Loki<br/>(log store)"]

    NodeExp --> Prometheus
    IntelGPU --> Prometheus
    Heartbeat --> Prometheus
    Promtail --> Loki

    subgraph GRAFANA["Grafana"]
        EdgeDash["Dashboard: Edge Fleet Health<br/>- Per-edge: CPU %, memory %, NPU FPS<br/>- Camera status grid (green/red/yellow)<br/>- Inference latency P95<br/>- MQTT message rate<br/>- Error camera list"]
        CloudDash["Dashboard: Cloud Health<br/>- MQTT consumer lag<br/>- TimescaleDB insert rate<br/>- Redis memory usage<br/>- WebSocket connection count<br/>- API response latency P95"]
    end

    Prometheus --> GRAFANA
    Loki --> GRAFANA

    Alertmanager["Alertmanager<br/>→ PagerDuty / Slack / email<br/><br/>Alerts:<br/>edge_offline: no heartbeat > 30s<br/>npu_drop_rate: dropped > 50/min<br/>memory_high: RSS > 90%<br/>disk_high: usage > 85%<br/>camera_error: STREAM_ERROR > 5 min<br/>mqtt_lag: consumer lag > 10s<br/>db_insert_fail: write errors > 0"]

    GRAFANA --> Alertmanager
```

### 9.9 Cloud Deployment

**Option A: Single Server (Docker Compose)**

```yaml
# docker-compose.yml
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports: ["1883:1883", "9001:9001"]
    volumes: ["./mosquitto.conf:/mosquitto/config/mosquitto.conf"]

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: airport_queue
      POSTGRES_USER: aqm
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes: ["pgdata:/var/lib/postgresql/data"]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  app:
    build: ./cloud
    environment:
      DATABASE_URL: postgresql+asyncpg://aqm:${DB_PASSWORD}@timescaledb/airport_queue
      REDIS_URL: redis://redis:6379
      MQTT_HOST: mosquitto
    ports: ["8000:8000"]
    depends_on: [mosquitto, timescaledb, redis]

  caddy:
    image: caddy:2
    ports: ["80:80", "443:443"]
    volumes: ["./Caddyfile:/etc/caddy/Caddyfile", "./dashboard/build:/srv"]

volumes:
  pgdata:
```

**Option B: Managed Services (AWS)**

| Component | AWS Service |
|---|---|
| FastAPI | ECS Fargate or EC2 |
| TimescaleDB | RDS PostgreSQL + TimescaleDB extension, or Timescale Cloud |
| Redis | ElastiCache Redis |
| MQTT | Amazon MQ (ActiveMQ with MQTT) or self-hosted Mosquitto on EC2 |
| Caddy/TLS | ALB with ACM certificate |
| Dashboard | S3 + CloudFront |

### 9.10 Cloud Server Failure and Disk Corruption

The cloud server is a single point of failure for the **dashboard and historical data**, but NOT for edge processing. Edges are fully autonomous.

**Scenario: Cloud server disk failure / data corruption**

```
Impact:
  - Dashboard goes down (operators lose visibility)
  - TimescaleDB data may be lost or corrupted (zone history, config, assignments)
  - Redis state lost (latest metrics cache)
  - MQTT broker down (edges buffer locally)
  - Edge processing: ZERO impact (continues running independently)

What edges do:
  - Continue decode, inference, tracking, analytics for all cameras
  - Buffer MQTT messages to local SQLite (100 MB per edge)
  - Use local config cache to maintain camera assignments and zone definitions
  - If edge restarts during cloud outage, local config cache lets it resume
```

**Prevention: Backups**

| Component | Backup Strategy | RPO (Recovery Point Objective) |
|---|---|---|
| TimescaleDB | pg_dump daily + WAL archiving to S3/NFS for continuous backup | Minutes (WAL) or 24h (dump) |
| Zone/camera/area config | Part of TimescaleDB backup. Also: export config as YAML, version in git. | Near-zero if git-tracked |
| Redis | Not backed up. It's a cache. Rebuilt from MQTT stream on restart. | N/A |
| MQTT broker | Stateless (or minimal state). Restart is sufficient. | N/A |
| Edge local config | Survives edge reboot. Refreshed from cloud on reconnect. | N/A |

**Recovery from total cloud loss:**

```
1. Provision new cloud server (or restore from VM snapshot)
2. Deploy Docker Compose stack
3. Restore TimescaleDB from backup (pg_restore)
   - Zone definitions, camera assignments, area config: restored
   - Historical metrics: restored up to backup point
   - Gap between last backup and failure: lost
4. Start MQTT broker
   - Edges detect broker is reachable
   - Edges drain their local SQLite buffers
   - Buffered metrics fill the gap (if outage < buffer capacity, ~55 min per edge)
5. Start FastAPI app
   - Reads camera assignments from restored DB
   - Pushes MQTT assign commands to all edges (edges already running, this confirms assignments)
   - Redis populates from live MQTT stream within seconds
   - Dashboard comes online
6. Total recovery time: 15-30 min (dominated by DB restore size)
```

**For higher availability (if the airport requires it):**

| Approach | Complexity | What It Protects |
|---|---|---|
| **Daily pg_dump + WAL to S3** | Low | Historical data, config. Loses last few minutes of metrics. |
| **PostgreSQL streaming replica** | Medium | Hot standby DB. Failover in <30s. Near-zero data loss. |
| **Two cloud servers (active-passive)** | Medium-High | Full redundancy. Passive server subscribes to same MQTT topics. Failover in <1 min. |
| **Managed services (RDS Multi-AZ, ElastiCache)** | Low (ops), Higher (cost) | AWS handles replication, failover, backups automatically. |

For most airport deployments, **daily pg_dump + WAL archiving** is enough. The real-time dashboard recovers from the live MQTT stream. Historical data is nice-to-have for trend analysis but not critical for real-time operations.

If the airport needs near-zero downtime for the dashboard, add a **PostgreSQL streaming replica** and a second app server behind a load balancer. The MQTT broker can run Mosquitto with bridge mode for replication, or use EMQX which has built-in clustering.

---

## 10. Degradation Matrix

| Failure | Detection | Impact | Dashboard | Edge | Data Loss | Recovery |
|---|---|---|---|---|---|---|
| **1 edge device down** | No heartbeat 30s | ~30 cameras / ~60 zones dark | STALE -> FAILOVER -> RECOVERING | Unaffected edges absorb cameras | Zero (historical). <=30s checkpoint staleness. | ~60s to metrics. ~120s to full accuracy. |
| **2+ edges down simultaneously** | Same per edge | 60+ cameras dark. Surviving edges may hit capacity. | Same. Low-priority zones may show "UNASSIGNED". | Remaining edges run near max. | Same per edge. | Longer -capacity-limited reassignment. |
| **Cloud server down** | Edges can't publish MQTT | Dashboard inaccessible. | Down. | **All edges continue processing.** Zero impact on inference. MQTT buffers locally (100 MB). | Zero if outage < ~55 min per edge. Oldest metrics dropped after buffer fills. | Buffer drains on reconnect. Dashboard auto-reconnects. |
| **Network: edge <-> cloud** | Same as cloud down for affected edge | Same as cloud down for affected edge | Those zones go STALE. | **Edge continues processing.** Buffers locally. | Same as cloud down. | Buffer drains when network returns. |
| **MQTT broker down** | Consumer fails to connect | All zones stale. | Shows last-known values with STALE badge. | All edges continue processing. Buffer locally. | Minimal (Redis retains last state for ~30s). | Broker auto-restarts (systemd). Buffer drain. |
| **TimescaleDB down** | Insert errors in consumer | Real-time dashboard works (Redis). Historical queries fail. | Real-time: OK. Charts/history: error. | No impact. | MQTT consumer buffers writes in Redis list (capped). | DB recovers, buffer drains. |
| **Redis down** | WebSocket gateway can't read | Dashboard stops receiving live updates. | Frozen. REST API still works (reads DB). | No impact. | None (Redis is cache, not source of truth). | Redis restarts, gateway reconnects. |
| **Single camera RTSP fails** | Edge reports in heartbeat | 1-4 zones dark (zones fed by this camera). | "STREAM ERROR" badge on affected zones. | Edge retries every 10s. Other cameras unaffected. | Metrics gap for affected zones only. | Automatic when camera stream recovers. |
| **NPU failure on edge** | Inference FPS drops to 0. Heartbeat shows npu_dropped spike. | All cameras on that edge produce no detections. | All zones from that edge go STALE. | Fallback to CPU inference (slower, may not sustain all cameras). | Metrics degrade or stop. | Replace edge / drain and reassign cameras. |
| **Cloud disk failure** | Server unresponsive or DB errors. | Dashboard down. Config/history at risk. | Down. | **All edges continue.** Buffer locally. Use local config cache. | Historical metrics since last backup. Config: zero if git-tracked. | Restore from backup. Edges drain buffers. 15-30 min. |
| **Cloud data corruption** | DB integrity errors, query failures. | Dashboard shows errors or wrong data. | Degraded or down. | **All edges continue.** | Depends on corruption extent. | Restore clean backup. Edges drain buffers. |
