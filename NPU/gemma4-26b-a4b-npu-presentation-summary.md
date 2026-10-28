# Gemma4-26B-A4B(MoE) NPU로 돌리기 (nntrainer PR #4428)

device : S26U (Hexagon v81) / 일부 S25U

| | 처음 (10/10) | 현재 |
|---|---|---|
| prefill 1024토큰 | 402 TPS | 627 TPS |
| prefill 2048토큰 | 398 TPS | 700 TPS |
| 첫 토큰 준비 | 1,463 ms | 506 ms |
| decode | 10~13 TPS | 13~15 TPS (2-bit / ternary pack) / 19.5 TPS (최신 pack) |
| peak 메모리 | - | 3.6~4.2 GB |

지난 발표 추후 계획 진행
- Gemma4-26B-A4B NPU 실행 → 완료
- MoE expert / projection / RMSNorm / RoPE / router / lm_head / attention NPU 연산 → 완료
- decode NPU 연산 → 완료
- CPU 연산 : tokenizer, embedding 조회만 남음
- SampleTestAPP에서 NPU 백엔드 선택 가능

---

## 1. 문제

- 모델 > NPU 메모리
  - weight 6.6~7.5 GB, NPU 매핑 한도 3.8 GB
  - expert는 layer당 16개만 상주, 나머지는 flash에서 읽음
- expert 수 : LFM2 대비 4배 (layer당 32 → 128개, top-8, 30 layer)
- NPU 연산 증가 → FastRPC 콜 / 복사 비용이 matmul보다 커짐

---

## 2. Weight 포맷 : 2-bit → ternary

pack 변천

| 시기 | expert | FC / head | embedding | pack 크기 |
|---|---|---|---|---|
| 10/13 QS2 pack | QS2CX_WH (2-bit 코드 + int4 4-level palette) | QS4CX (int4 + 열별 scale) | Q4_0 | 7.5 GB |
| 10/15 TL2 pack | TL2CX_WH (ternary -1/0/1, 1.67 bit) | QS4CX_WH_AFFINE (int4 + scale + offset) | Q4_0 | 6.6 GB |
| 10/16 rotated | TL2 + online Hadamard 회전 | affine | Q4_0 | |
| 10/20 strict | TL2 + Paley-I 회전, static activation 사이드카 | affine | U4 asym | |

ternary (TL2CX_WH)
- 값 3개(-1, 0, 1) → (24 × 128) weight 블록을 640 byte에 pack → 1.67 bit/weight
- 레코드 = packed 코드 + N개 fp32 scale + N개 fp32 column sum (palette, scale/7 없음)
- prefill : VTCM에서 int4 WH로 펼쳐서 HMX
- decode : packed 그대로 HVX LUT GEMV (펼치지 않음, 읽는 바이트 그대로 절반 이하)
- 28/29층은 recipe상 4-bit affine 컨테이너 → 값은 ternary라 전부 TL2로 바꿔도 동일
  - arena 2,624 → 1,792 MiB, peak 4,723 → 3,669 MB, prefill 577 → 613 TPS, decode 11.8 → 15.5 TPS

affine (QS4CX_WH_AFFINE)
- w = scale × (q + offset) → BF16 원본 값을 그대로 복원 (대칭 min/max 재양자화 없음)
- kernel이 offset × Σ(u8 - zp) 보정을 accumulator에 더함

expert 1개 크기 (gate_up 2816 → 1408, down 704 → 2816)

| 포맷 | gate_up | down | slot |
|---|---|---|---|
| BF16 | 7.56 MiB | 3.78 MiB | 11.3 MiB |
| int4 affine | 1.89 MiB | 0.95 MiB | 3.0 MB |
| 2-bit QS2 | 0.95 MiB | 0.47 MiB | 1.45 MiB |
| ternary TL2 | 811 KiB | 413 KiB | 1.23 MiB |

- decode 토큰당 expert 240개 읽음 → QS2 349 MiB / TL2 295 MiB
- 참고 LFM2 expert (int4) 5.25 MiB

WH 레이아웃
- 변환기가 HMX 타일 순서(WH)로 미리 써 둠 → load는 읽기만
- load 때 변환하면 157 MB/s → 모델 하나에 25 s

rotated checkpoint
- weight가 미리 Hadamard 회전된 상태 → 런타임이 expert 입력에 같은 회전 적용 (gate/up H44×H64, down H44×H16)
- 회전 sign은 rotation_signs.json (seed에서 재생성, 2816 + 704개)
- decode HVX FP32 / prefill HMX FP16
- strict pack은 Paley-I 기저 → sign을 scalar로 곱하던 초기 구현이 prefill 353 TPS → HVX화로 683.70 TPS

변환기
- weight_converter.py : BF16 HF checkpoint → QS2 pack → convert_tl2_model → affine FC/head → rotation_signs.json 한 번에
- 50 GB safetensors를 mmap 대신 pread (torch 불필요)

---

## 3. Prefill

콜 수 줄이기 (fusion)
- q/k/v → qkv 1콜, dense FFN 3개 → 1콜, norm / residual add는 앞 matmul 콜 안으로
- layer당 NPU 콜 7 → 2
- 행 분할을 CPU가 아니라 NPU(skel) 안에서
  - 기존 : VTCM에 안 들어가면 CPU가 2~4콜로 나눠 보냄 (콜마다 버퍼 크기 비례 고정비)
  - 변경 : 1콜로 보내고 NPU가 내부에서 나눠 처리
- S25U 1024토큰 : 4.8 s → 3.8 s

복사 줄이기
- 기존 : 콜 끝날 때마다 결과를 CPU로 복사 → 다음 콜에서 다시 ION으로 복사
- 변경 : 블록 안 중간 결과는 NPU 쪽 버퍼에 유지
- 2048토큰 복사 시간 402 → 39 ms

MoE kernel
- 행 quantize 1번만 (기존 : expert 8개로 가면 8번) → 2048토큰 4.6 → 4.0 s
- padding만 있는 구간 pack skip → 2048토큰 3.2 → 3.0 s
- HMX 연산 중에 HVX가 앞 결과 후처리 (dequant, GeGLU, scatter 중첩)

Attention
- int8 attention kernel : score → softmax까지 VTCM 안에서 처리
- 4096토큰 이상에서 fp16 경로로 떨어지던 문제 제거 → 2.8만 토큰 prefill 69.5 → 39.9 s

| prefill (S26U) | 1024토큰 | 2048토큰 |
|---|---|---|
| 처음 (10/10) | 2.55 s (402 TPS) | 398 TPS |
| 현재 | 1.63 s (627 TPS) | 2.92 s (700 TPS) |

---

## 4. Decode

구조 : DSP 상주 그래프
- decode step 전체(542 op)를 NPU에 그래프로 상주
- CPU → position 전달 / NPU → 다음 token id만 반환
- 토큰당 메시지 1번, 그 사이 CPU idle

첫 토큰 준비 : KV cache seed + 그래프 초기화
- prefill과 decode는 attention 커널이 다름 → KV cache도 따로 (레이아웃, scale 단위 다름)

| | prefill attention | decode attention (ATTN_M1) |
|---|---|---|
| 연산 | HMX, 64행 블록 | HVX, 행 1개 |
| cache | HMX 타일 순서 int8, scale 층당 1개 | head별 행 순서 int8, scale (층, head, dim)당 1개 |

- seed = prefill이 끝난 prompt의 K/V(host fp16 사본)를 decode cache에 옮겨 담는 작업
  - 층마다 : fp16 행 읽기 → (head, dim)별 |max|로 scale 결정 → int8 양자화 → append
  - 처음 : 첫 토큰 3.0~3.2 s 중 seed 2.3~2.5 s (host fp16→f32 변환 0.3 s + DSP 스레드 1개가 양자화 2.0 s)
- 줄인 것
  - |max| / append를 DSP worker pool + HVX 벡터화 : seed RPC 429 → 93 ms
  - fp16 비트 그대로 전송, DSP에서 넓힘 : host 변환 145 → 0 ms, 전송 바이트 절반
  - host cache(rpcmem)를 복사 없이 제자리에서 읽기
  - RoPE 테이블 층마다 전송(30번) → 이름으로 공유(1번) : 439 ms 제거
- 첫 토큰 1,463 → 506 ms (p1024), 2,206 → 636 ms (p2048)
- seed 후 host fp16 사본 해제 → +470~520 MiB, expert pool이 사용

Attention
- KV cache int8 + 정수 연산 attention
- 토큰당 attention 20.9 → 6.7 ms (S25U → S26U)

expert cache (decode 시간 대부분 = expert weight 읽기)
- 첫 토큰 후 남는 메모리로 상주 expert 16 → 최대 34개 : 10.4 → 13.1 TPS
- expert 교체 정책 LRU → LRFU
- 다음 layer expert 예측 prefetch (보조, 단독 효과 작음)
- weight 2-bit → ternary(1.67 bit) : 읽는 양 감소, 11.8 → 15.5 TPS

---

## 5. 메모리

```
VTCM 8 MiB (prefill MoE)          VTCM 8 MiB (decode MoE)
├─ 입력 64행        176 KiB       ├─ expert 버퍼 A   968 KiB  ← 짝수 expert
├─ gate_up weight   1.89 MiB      ├─ expert 버퍼 B   968 KiB  ← 홀수 expert
├─ down weight      0.95 MiB      │   한쪽 계산 중 다른 쪽에 다음 expert DMA
├─ 중간/결과 버퍼    1.8 MiB       └─ 나머지 여유
└─ 합계 4.8 MiB
```

expert 1개 크기는 2장 표 참고

| 항목 | 전 | 후 |
|---|---|---|
| 3만 토큰 KV cache | 6.5 GB | 495 MB |
| sliding 층 decode cache | 640 MiB | 106 MiB |
| expert 전부 ternary | peak 4.7 GB | 3.7 GB |
| lm_head NPU (CPU 사본 제거) | 396 MiB | 0 |

---

## 6. GPU 비교 (같은 칩, weight 포맷 다름)

| | prefill 1K | decode | peak |
|---|---|---|---|
| GPU (Adreno 840) | 388 TPS | 20~25 TPS | 3.4~7.8 GB |
| NPU | 627 TPS | 13~15 TPS (최신 pack 19.5) | 3.6~4.2 GB |

- prefill / 메모리 : NPU 우위
- decode : GPU가 빠르거나 비슷

---

## 추후 계획

- decode 16 TPS (기존 pack 미달, 최신 pack 달성)
- 긴 문맥 decode : 3만 토큰에서 attention이 토큰 시간의 51%
- static 양자화 모드 속도 개선 (현재 prefill 82 TPS)
- PetCare 에이전트 시나리오 tool 호출 정확도

---

## 부록. 상세 설명

### A. 행 분할을 NPU(skel) 안에서 처리

FC 콜 하나가 VTCM에 올릴 수 있는 행 수
- VTCM = 입력 행(u8) + weight slice double buffer + staging 512 KiB
- 실제 값 : qkv(K=2816) 1,344행 / o proj(K=4096) 1,472행 / full 층 o proj(K=8192) 704행
- 2048행 prefill이면 qkv 2번, o proj 2~3번 나눠야 함

```
기존 : CPU가 나눠서 여러 번 호출
CPU ── call(행 0~1343) ──▶ NPU   ← 버퍼 전체 map/sync (qkv 4.3 ms)
CPU ── call(행 1344~2047) ─▶ NPU  ← 또 버퍼 전체 map/sync

변경 : CPU는 1번 호출, NPU가 내부 loop
CPU ── call(행 0~2047) ──▶ NPU   ← map/sync 1번
                             for (m0 = 0; m0 < 2048; m0 += 1344)
                                 행 m0 ~ m0+1343 계산 → 출력의 같은 행 위치에 바로 씀
```

- norm / RoPE는 행 단위 연산이라 나눠 계산해도 결과 동일
- FastRPC map/sync 비용 : 실제 사용 길이와 무관하게 버퍼 MiB당 약 27 us
- p2048 : qkv 65 → 35콜 (638 → 506 ms), o proj 95 → 60콜 (410 → 291 ms)

### B. 행 quantize 1번만 (MoE pack)

MoE 콜 입력 : f32 행 M개, 행마다 expert 8개로 routing
HMX 입력 : expert별로 자기 행만 모아서 u8 타일(AH)로 pack

```
기존 : expert마다 f32에서 다시 quantize
expert 3  : row 7 (f32 읽기 → scale 계산 → u8 변환 → pack)
expert 15 : row 7 (f32 읽기 → scale 계산 → u8 변환 → pack)   ← 같은 행 반복
...         row 7이 8번 quantize됨

변경 : quantize는 1번, expert별로는 byte만 모음
1) scan : 모든 행을 1번씩 u8로 quantize (행 순서대로 버퍼에 저장)
2) pack : expert마다 자기 행의 u8 byte만 gather → AH 타일
```

- quantize는 행 단위 scale이라 어느 expert로 가도 같은 byte → 공유 가능 (결과 bit 동일)
- 같은 커밋에서 down 결과 scatter도 타일 단위 → 행 단위 연속 쓰기로 변경
- 2048행 MoE 콜 worker 시간 : pack 63 → 27 ms, down scatter 128 → 74 ms
- prefill p2048 4,485~4,970 → 3,969~4,109 ms

### C. HMX 연산 중에 HVX가 앞 결과 후처리

용어
- GU : gate_up matmul (HMX) / DN : down matmul (HMX)
- DQ : dequant + GeGLU (HVX) / RQ : down 입력용 requant (HVX) / SC : down 결과 dequant + scatter (HVX)

```
기존 : 직렬 (HMX가 HVX 끝날 때까지 대기)
HMX │█GU b0█│      │█GU b1█│      │█GU b2█│      │
HVX │       │▒DQ b0▒│      │▒DQ b1▒│      │▒DQ b2▒│
                ↑ HMX idle       ↑ HMX idle

변경 : 결과 버퍼(acc staging) 2개 A/B 교대
HMX │█GU b0→A█│█GU b1→B█│█GU b2→A█│█GU b3→B█│
HVX │         │▒DQ b0 A▒│▒DQ b1 B▒│▒DQ b2 A▒│▒DQ b3 B▒│
              HMX가 B를 채우는 동안 HVX가 A를 처리
```

64행 블록 단위로도 한 칸씩 밀어서 겹침

```
HMX │GU n│GU n+1│DN n │GU n+2│DN n+1│
HVX │    │DQ n  │RQ n │SC n  │DQ n+2 ...
     → 블록 n의 down을 블록 n+1의 gate_up 뒤에 발행 → 블록 n의 RQ가 HMX 연산 뒤에 숨음
```

weight DMA도 겹침

```
DMA │GU w(e)│DN w(e)│GU w(e+1) → gate_up 영역 재사용│
HMX │       │██ GU e ██│██ DN e ██│██ GU e+1 ██│
```

효과
- (LFM2) 한쪽 유닛이 놀던 시간 6.9 ms / 22.2 ms 콜 → quant 1,440 → 354 us, scatter 1,056 → 23 us
- (Gemma4, S25U) down epilogue를 background로 : 콜 32.2 → 30.8 ms, scatter 대기 2.8 → 0.3 ms
- (Gemma4 2-bit, S26U) down epilogue 대기열 분리 : 대기 13.5 → 4.4 ms, 콜 30.6 → 26.1 ms

### D. Attention을 NPU로 처리한 효과

| 비교 | 전 | 후 |
|---|---|---|
| 1.9만 토큰 prefill 전체 (S26U) | CPU attention 423 s | NPU attention 약 49~50 s (385~394 TPS) |
| 1024토큰, 30 layer attention 시간 합 (S25U) | 774 ms | 425~497 ms |
| 2.8만 토큰 prefill (4096열 이상 fp16 fallback 제거) | 69.5 s | 39.9 s |
| decode 토큰당 attention (S25U → S26U) | 20.9 ms | 6.7 ms |

- 1.9만 토큰 : 같은 기기에서 attention만 CPU로 돌린 기준 실행과 비교, 약 8.5배 (49~50 s는 TPS로 환산한 값)
- 1024토큰 774 ms는 이미 NPU 실행 상태이고, 그 뒤 host 쪽 비용 정리(KV 등록 load로 이동, append 양자화 HVX 병렬화)로 줄인 것
- kernel 단독 : sliding 층(1024 × 1024) 1개 18.4 ms
- 짧은 문맥(1K)에서 CPU attention vs NPU attention 직접 비교한 수치는 PR에 없음
