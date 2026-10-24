# Federated Learning Exchange Protocol Specification (v2.0)

**Project:** Federated Learning Pipeline on Heterogeneous Edge Devices  
**Author:** Farhan (Software Pipeline Lead)  
**Hardware Partner:** Niksh (AMD Kria Lead)  
**Supervisor:** Priyadarshan  
**Lab:** Dixon IoT Lab, Plaksha University  
**Date:** October 2026  
**Document Status:** Week 1 Deliverable — Production-Grade Specification  

---

## 1. Executive Summary & Design Principles

The goal of this project is to orchestrate federated learning rounds between two heterogeneous edge devices:
1. **AMD Kria K26/KV260 board** (Quad ARM Cortex-A53 running Linux / Python / PyTorch)
2. **Microcontroller** (ESP32-P4 running embedded C/C++ firmware with ESP-DL)

### Core Architectural Decisions:
* **Embedded-Safe Wire Format:** Discard Python `pickle` and bulky JSON float arrays. The protocol uses **JSON Metadata** for structural descriptors and **Raw Contiguous Little-Endian Binary Buffers** for tensor values.
* **Separation of Concerns:** Model download is simplified into dedicated manifest and binary endpoints to avoid requiring MIME multipart parsers in C on the microcontroller.
* **Clean Evaluation Separation:** Inference-only / evaluation reports are handled by a distinct endpoint (`/evaluation/report`), keeping training sample counts and FedAvg weight calculations unpolluted.
* **Memory Safety & Alignment:** Explicit validation rules prevent buffer overflows, and C/C++ clients use safe memory copies (`memcpy`) to avoid hardware unaligned memory faults on RISC-V / ARM architectures.
* **Deterministic Concurrency:** Formal rules govern stale updates, duplicate submissions, client timeouts, and state transitions.

---

## 2. System Architecture & Topology

The topology is a **star network** where the laptop acts as the central coordinating parameter server over a local TCP/IP network (Wi-Fi or Ethernet).

```
       ┌────────────────────────────────────────────────────────┐
       │              Central Aggregator Server                 │
       │                   (Farhan's Laptop)                    │
       │            FastAPI Parameter Server (Port 8000)        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                     HTTP/1.1 REST over Local Network
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌───────────────────────────────┐           ┌────────────────────────────────┐
│         Client Device 1       │           │        Client Device 2         │
│         AMD Kria Board        │           │       ESP32-P4 Board           │
│  (ARM Cortex-A53 / Linux)     │           │  (Dual RISC-V / FreeRTOS)      │
│  - Python / PyTorch runtime   │           │  - Embedded C/C++ runtime      │
│  - Trains Backbone + Head     │           │  - Evaluates / Trains Head     │
└───────────────────────────────┘           └────────────────────────────────┘
```

---

## 3. Communication Endpoints (REST API Specification)

The parameter server exposes the following endpoints under `http://<server-ip>:8000/api/v1`.

### 3.1. `GET /api/v1/round/status`
Allows client devices to check the current orchestrator status before downloading weights or uploading updates.

* **Query Parameters:**
  * `device_id` (string, required): e.g. `"amd-kria-01"` or `"esp32-p4-01"`
* **Response (JSON):**
  ```json
  {
    "status": "ROUND_IN_PROGRESS",
    "current_round_id": 1,
    "model_version": 1,
    "timeout_remaining_seconds": 180,
    "min_clients_required": 2
  }
  ```
* **State Machine Values:**
  * `WAITING_FOR_CLIENTS`: Server initialized, waiting for devices to poll.
  * `ROUND_IN_PROGRESS`: Round active; server waiting for training updates.
  * `AGGREGATING`: Aggregator is running FedAvg; uploads temporarily blocked.
  * `ROUND_COMPLETED`: Current round finished; new global model ready.
  * `FAILED`: Round aborted due to timeout or unrecoverable error.

---

### 3.2. Model Download Endpoints (C-Friendly Design)
To prevent microcontrollers from needing complex MIME multipart parsers, model distribution is decoupled into two clean calls:

#### A. `GET /api/v1/model/manifest`
Fetches structural descriptors and tensor layouts for the current global model.
* **Query Parameters:**
  * `device_id` (string, required)
  * `requested_layers` (string, optional): `"all"` (default for Kria) or `"head_only"` (for ESP32).
* **Response (JSON):**
  ```json
  {
    "message_type": "global_model_manifest",
    "round_id": 1,
    "model_version": 1,
    "total_payload_bytes": 73856,
    "tensor_manifest": [
      {
        "name": "head.cls_conv.weight",
        "shape": [32, 64, 3, 3],
        "dtype": "float32",
        "offset_bytes": 0,
        "size_bytes": 73728
      },
      {
        "name": "head.cls_conv.bias",
        "shape": [32],
        "dtype": "float32",
        "offset_bytes": 73728,
        "size_bytes": 128
      }
    ]
  }
  ```

#### B. `GET /api/v1/model/weights`
Streams the contiguous raw binary byte buffer of parameters.
* **Query Parameters:**
  * `device_id` (string, required)
  * `model_version` (integer, required): Ensures the binary payload strictly matches the downloaded manifest.
* **Response:** Raw binary octet stream (`Content-Type: application/octet-stream`).

*(Alternative single-request mode: The server can also return the binary stream directly with the manifest embedded in a custom HTTP header `X-Model-Manifest: <JSON-string>`)*.

---

### 3.3. `POST /api/v1/model/upload` (Training Updates)
Used exclusively by clients submitting **trained weight updates** for aggregation.

* **Request Format:** HTTP `multipart/form-data` with two parts:
  * **Part 1: `metadata` (`application/json`)**:
    ```json
    {
      "message_type": "client_update",
      "device_id": "esp32-p4-01",
      "round_id": 1,
      "base_model_version": 1,
      "num_samples": 80,
      "updated_layers": ["head"],
      "metrics": {
        "training_loss": 0.384,
        "training_duration_ms": 6200
      },
      "tensor_manifest": [
        {
          "name": "head.cls_conv.weight",
          "shape": [32, 64, 3, 3],
          "dtype": "float32",
          "offset_bytes": 0,
          "size_bytes": 73728
        },
        {
          "name": "head.cls_conv.bias",
          "shape": [32],
          "dtype": "float32",
          "offset_bytes": 73728,
          "size_bytes": 128
        }
      ]
    }
    ```
  * **Part 2: `weights` (`application/octet-stream`)**:
    * Contiguous byte buffer containing updated `float32` parameters matching `tensor_manifest`.

* **Server Response (JSON):**
  * `200 OK`: `{"status": "ACCEPTED", "message": "Update validated for round 1"}`
  * `200 OK (Duplicate)`: `{"status": "DUPLICATE_IGNORED", "message": "Device update already recorded"}`
  * `409 Conflict (Stale)`: `{"status": "REJECTED", "reason": "STALE_VERSION"}`
  * `409 Conflict (Locked)`: `{"status": "REJECTED", "reason": "AGGREGATION_IN_PROGRESS"}`
  * `422 Unprocessable`: `{"status": "REJECTED", "reason": "MANIFEST_PAYLOAD_MISMATCH"}`

---

### 3.4. `POST /api/v1/evaluation/report` (Inference / Evaluation Only)
Dedicated endpoint for edge devices performing **evaluation only** (e.g. testing global model accuracy on live camera feed without backpropagation).

* **Request (JSON):**
  ```json
  {
    "message_type": "evaluation_report",
    "device_id": "esp32-p4-01",
    "round_id": 1,
    "evaluated_model_version": 1,
    "eval_samples": 120,
    "metrics": {
      "validation_loss": 0.412,
      "mean_iou": 0.68,
      "avg_latency_ms": 967,
      "fps": 1.03
    }
  }
  ```
* **Server Response:**
  ```json
  {
    "status": "RECORDED",
    "message": "Evaluation metrics logged for model version 1"
  }
  ```

---

## 4. Binary Safety & Memory Alignment Contract

To guarantee cross-platform safety between x86 (Laptop), ARM Cortex-A53 (Kria), and RISC-V (ESP32-P4):

### 4.1. Server-Side Safety Validations
Before accepting any upload, the server strictly validates:
1. **Size Integrity:** For every tensor in `tensor_manifest`:
   $$\text{size\_bytes} == \left(\prod \text{shape}\right) \times \text{sizeof}(\text{dtype})$$
2. **Bounds Check:** For every tensor:
   $$\text{offset\_bytes} + \text{size\_bytes} \le \text{total\_payload\_bytes}$$
3. **No Overlap:** Sort entries by `offset_bytes`. Ensure $\text{offset}_i + \text{size}_i \le \text{offset}_{i+1}$.
4. **Payload Match:** The actual byte count received in the HTTP request body must exactly equal `total_payload_bytes`.
5. **Consistency:** All layer names in `updated_layers` must be represented in `tensor_manifest`.

### 4.2. Embedded Memory Alignment & Safe Copying (ESP32-P4)
* Directly casting a raw byte pointer `(float*)(buffer + offset)` can trigger a **hardware CPU exception (unaligned memory access)** on embedded architectures if the offset is not 4-byte aligned.
* **Firmware Rule:** Always use `memcpy` to copy tensor parameters into 32-bit aligned model memory buffers:
  ```c
  // Safe C/C++ memory transfer
  memcpy(layer_weights_ptr, binary_payload + manifest_item.offset_bytes, manifest_item.size_bytes);
  ```

---

## 5. Parameter Compatibility Rules for FedAvg

FedAvg cannot combine arbitrary tensors. Two devices' parameters are considered **compatible for averaging** if and only if they meet all 4 conditions:

1. **Exact Name Match:** Parameter names must be identical (e.g. `head.cls_conv.weight`).
2. **Identical Dimension / Shape:** The tensor dimensions must match exactly (e.g. `[32, 64, 3, 3]`).
3. **Matching Data Type:** Both must be `float32`.
4. **Architectural Semantics:** The layers must have the same receptive field, stride, and role in the network.

**Rule for Non-Compatible or Missing Layers:**
* Parameters trained by only one device (e.g. Backbone trained only on Kria) are **not averaged**; the global model directly adopts the trained parameters from that device.
* Only mutually trained layers that pass compatibility checks are aggregated via weighted FedAvg.

---

## 6. Heterogeneous Aggregation Mathematics

Let $\mathcal{K}$ be the set of reporting clients. Each client $k \in \mathcal{K}$ reports $n_k$ local training samples.

### 6.1. Head Aggregation (Mutually Trained Layers)
For layers present in both Kria and ESP32 uploads:
$$W_{\text{head}}^{t+1} = \frac{n_{\text{kria}} W_{\text{head}}^{\text{kria}} + n_{\text{esp}} W_{\text{head}}^{\text{esp}}}{n_{\text{kria}} + n_{\text{esp}}}$$

### 6.2. Backbone Aggregation (Kria-Only Update)
For layers trained solely by Kria (due to ESP32 memory constraints):
$$W_{\text{backbone}}^{t+1} = W_{\text{backbone}}^{\text{kria}}$$

### 6.3. Inference-Only Device Behavior
Devices posting to `/evaluation/report` contribute $n_{\text{eval}}$ samples for validation tracking, but **0 samples to the training average** ($n_k = 0$). The global model updates solely from training contributors.

---

## 7. State Machine & Concurrency Rules

```
Laptop Server                                    Edge Client (Kria / ESP32)
      │                                                      │
      │ 1. GET /api/v1/round/status                          │
      │ ◄─────────────────────────────────────────────────── │
      │ ─── status: ROUND_IN_PROGRESS ─────────────────────► │
      │                                                      │
      │ 2. GET /api/v1/model/manifest & /weights             │
      │ ◄─────────────────────────────────────────────────── │
      │ ─── [JSON Manifest + Pure Binary Weights] ─────────► │
      │                                                      │
      │                                          ┌───────────┴───────────┐
      │                                          │ Local Work:           │
      │                                          │ Train or Evaluate     │
      │                                          └───────────┬───────────┘
      │                                                      │
      │ 3. POST /api/v1/model/upload (or /evaluation/report) │
      │ ◄─── [Metadata + Contiguous float32 Buffer] ──────── │
      │                                                      │
      │ [Server validates bounds & manifests]                │
      │ ─── 200 OK (ACCEPTED) ─────────────────────────────► │
      │                                                      │
┌─────┴────────────────────────┐                             │
│ Aggregation Trigger:         │                             │
│ All required clients submit  │                             │
│ OR timeout timer expires     │                             │
│ Compute FedAvg               │                             │
│ Persist checkpoint to disk   │                             │
│ Increment round_id & version │                             │
└─────┬────────────────────────┘                             │
      │                                                      │
```

### 7.1. Concurrency Edge Cases
1. **Duplicate Submissions:** If device $k$ submits multiple uploads within the same round $t$, the server ignores the second upload (`200 OK: DUPLICATE_IGNORED`) to prevent skewing the average.
2. **Uploads Arriving During Aggregation:** If an upload arrives while the server is calculating FedAvg, the server responds with `409 Conflict: AGGREGATION_IN_PROGRESS`.
3. **Stale Submissions:** If an upload's `base_model_version` is less than the active `model_version`, the server rejects it (`409 Conflict: STALE_VERSION`).
4. **Client Straggler / Timeout Fallback:**
   * Each round runs a countdown timer (`ROUND_TIMEOUT_SECONDS = 300`).
   * **If at least 1 valid training update is received (e.g. Kria):** The server logs a timeout warning for the missing device, aggregates available updates, advances the round, and writes a recovery checkpoint.
   * **If 0 training updates are received:** The server marks the round as `FAILED` and resets.
5. **State Advancement & Persistence:**
   * After aggregation succeeds, the new global weights are written to disk as `models/checkpoints/round_{t+1}.pt`.
   * Only after the checkpoint file is safely synced to disk do `round_id` and `model_version` atomically increment ($t \to t+1$).
