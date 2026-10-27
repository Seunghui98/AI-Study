# Gemma4-26B-A4B(MoE) NPU로 돌리기 (nntrainer PR #4428)

device : S26U (SM8850, Hexagon v81) / 일부 S25U (v79)
수치 : PR 커밋 메시지, docs에 기록된 값

지난 발표 추후 계획 진행 상황
- google/gemma-4-26B-A4B-it 돌리기 → 완료 (prefill, decode 모두 NPU)
- MoE expert FFN + projection 연산 지원 → 완료
- RMS norm / RoPE / lm_head / router NPU 연산 지원 → 완료 (embedding만 CPU)
- attention core 연산 → 완료 (prefill int8 A8W8, decode int8 KV)
- decode 관련 작업 → 완료 (DSP 상주 그래프, 토큰당 메시지 1번)
- QuickDotAI API / SampleTestAPP에서 NPU 백엔드로 선택 가능

---

## 1. 모델 구조와 문제

Gemma4-26B-A4B
- 30 layer, layer당 expert 128개 중 8개 선택
- sliding attention 25층 (window 1024, KV head 8 × 256) / full attention 5층 (KV head 2 × 512)
- hidden 2816, expert intermediate 704
- vocab 262,144 → lm_head 하나가 352 MiB

expert
- 1개 = 1.45 MiB (2-bit QS2) / 1.23 MiB (ternary TL2) (자세한 건 4절 expert 1개 크기)
- 30 layer × 128 = 3,840 expert (≈ 5.4 GiB, QS2 slot 기준 계산값)
- pack 전체 7.5 GB (QS2) / 6.6 GB (TL2)

문제
- DSP가 매핑할 수 있는 총량(PD 한도) 3,840 MiB → expert 전부 못 올림
  - layer당 16개만 상주(C=16), 나머지는 flash에서 읽음
- LFM2 대비 expert 수 4배 (32 → 128)
- 지난번에 CPU가 하던 연산(attention, norm, router, RoPE, lm_head)까지 NPU로 → FastRPC 콜 / 복사가 늘어남
- 긴 context에서 KV cache를 층마다 max_seq_len만큼 잡으면 메모리 부족

---

## 2. Prefill

연산 위치
- NPU : qkv proj(+q/k norm, RoPE), attention, o proj(+residual), router(+norm, softmax, top-k), MoE FFN(+GeGLU, post norm, residual), dense FFN, lm_head 마지막 행(untied head일 때)
- CPU : tokenizer, embedding(Q4_0, 필요한 행만)

### 1) FastRPC 콜 줄이기 (fusion)

- q/k/v 3개 → qkv 1콜, dense MLP FC 3개 → 1콜 (GeGLU는 kernel epilogue)
  - layer당 HTP 콜 7 → 2
- RMSNorm 9개 중 7개를 matmul 콜 안으로
  - 기존 : DSP 콜 사이마다 CPU가 activation 전체를 다시 훑음 (512토큰에서 약 300 ms)
- o proj + residual_add / MoE + post-FFN norm + add 를 각각 한 콜로
  - 각각 30콜 감소, staging 7.2 → 6.2 GB, 5.56 → 4.9 GB
- skel 안에서 VTCM 크기만큼 행을 나눠서 돌기
  - p2048 qkv 65 → 35콜, o proj 95 → 60콜

S25U, 1024토큰 : 4,791 ms → 3,808 ms

### 2) 복사 줄이기

- 블록 안 텐서를 DSP staging(ION)에 그대로 유지
  - 기존 : 콜 끝날 때마다 host로 복사 → 다음 콜에서 다시 ION으로 복사
  - memcpy p2048 402 → 39 ms (7,444 → 408 MB)
- q/k/v slice를 출력 위치에 바로 씀
  - memcpy 264,000회 → 2,970회 (132.8 → 97.5 ms)
- staging 버퍼를 2^n 크기 대신 실제 텐서 크기로
  - FastRPC cache 유지가 버퍼 전체에 걸려서 11.5 MB 텐서가 16 MiB 비용을 냄
- 양자화 KV cache 등록을 prefill 밖(load)으로 → 1024행 prefill에서 124 ms 제거

### 3) MoE kernel

- 행 quantize 1번만
  - 기존 : 행 하나가 expert 8개로 가면 8번 quantize
  - 변경 : 1번 quantize, pack은 byte gather
  - p2048 4,485~4,970 → 3,969~4,109 ms
- padding slot만 있는 16행 단위는 pack skip (1024토큰에서 36 MB/콜), 출력 zero-fill 제거
  - 2048토큰 3,166 → 2,954 ms
- 2-bit expert는 VTCM에서 int4로 펼쳐서 HMX
  - 펼치기 job 뒤에 down epilogue가 줄 서던 것 → down epilogue는 foreground lane으로
  - DRAIN 13.5 → 4.4 ms, p2048 398 → 426 TPS
- HMX / HVX 동시 가동 : HMX가 다음 batch 도는 동안 앞 batch의 dequant, GeGLU, scatter를 HVX worker가 처리

### 4) QKV / o proj

- q/k head norm + RoPE를 matmul dequant epilogue 안에서
  - 기존 : matmul → 90 MB qkv 출력 → norm/RoPE로 한 번 더 왕복
  - p2048 3,442~3,582 → 3,288~3,325 ms
- Q를 f32로 바로 받아서 DSP에서 u16 quantize, context(u16)는 o proj에 바로 넘김
  - p1024 2,514 / 2,254 → 2,201 / 2,188 ms
- pre-norm scratch를 M × K 대신 lane당 4행만 (15~24 MB → 0.6 MB) → p4096 OOM 해결

### 5) Attention

- row-blocked A8W8 kernel
  - head 하나의 K^T / V를 VTCM 상주
  - 64행 블록마다 HMX QK^T → int16 readout → integer softmax → P'V
  - score가 VTCM 밖으로 안 나가고 재양자화 없음
  - sliding 층(1024 × 1024) : HMX 2.76 → 1.27 ms, softmax 7.5 → 2.8 ms
- V scale을 32-dim 그룹당 1개 → 행당 1개 : P quantize 8번 → 1번, 22,377 → 19,926 us
- HMX 곱을 inline 발행 (라이브러리 호출 약 50 cycle vs 실제 HMX 9 cycle)
- int8 calibration : CPU 단일 스레드 → DSP worker pool
- 4096열 넘으면 fp16으로 떨어지던 것 → segment로 나눠 계산 후 log-sum-exp로 merge
  - 22,241 cached + 6,048토큰 prefill 69.5 → 39.9 s

### 6) Router

- router norm을 logits kernel 안으로 + gate weight를 VTCM에서 읽기 : 195 → 156 ms (p1024, 30콜)
- top-k : 128개를 13번 스캔 → 정렬 삽입 1번 : 267 → 213 ms
- qf32 누적 NaN 버그 수정 (4행 중 3행 NaN → expert 0~7만 선택되던 문제, nll 12.33 → 3.536)

### 7) expert read-ahead (flash → DRAM)

- prefill은 예측 필요 없음 (1024행이면 거의 모든 expert 사용) → 남은 layer expert 전부 미리 읽기
- reader thread 4 → 6 : 1090토큰 1,747 → 1,698 ms
- 1024토큰 기준 expert 3,712개 읽음, 노출된 대기 70 ms / 1,752 ms

### 8) TL2 pack 전환 후 느려진 경로

- TL2 펼치기 / affine 보정이 VTCM을 scalar로 접근 → 벡터화
  - p2048 3,191 ms (641.81 TPS)
- Hadamard 회전(Paley) sign을 scalar 루프로 곱하던 것 → HVX
  - 2K 요약 353 → 683.70 TPS

prefill 결과 (S26U)

| | 1024토큰 | 2048토큰 |
|---|---|---|
| 처음 (10/10) | 2,549 ms (402 TPS) | 398 TPS |
| fusion + MoE kernel (10/11) | 1,962~2,023 ms | 3,969~4,109 ms |
| epilogue (10/11) | 1,718~1,943 ms | 3,288~3,325 ms |
| padding skip, staging (10/14) | 1,632 ms (627 TPS) | 2,954 ms (693 TPS) |
| reader 6 (10/16) | 1090토큰 634~650 TPS | 699~700 TPS |
| strict pack (10/20) | | 683.70 TPS |

---

## 3. Decode

### 1) 구조 : DSP 상주 그래프

- 기존 (LFM2) : MoE 세션 / FC 세션 2개 → 세션 간 hop wakeup 3.0~3.8 ms/token
- 변경 : 토큰 step 전체(542 op)를 DSP에 그래프로 상주, 세션 1개 (one PD)
  - ARM은 position만 mailbox에 씀 → DSP가 그래프 실행 → argmax token id만 반환
  - logits(262,144 × 4 B = 1 MiB) 안 받음, sampling 필요할 때만 요청
- expert miss도 토큰 안에서 처리
  - DSP가 miss 목록을 올림 → 가진 expert부터 계산 → 그동안 ARM이 flash에서 빈 slot으로 읽음 → DSP가 제자리 rebind
- prefill / decode가 FC weight 한 벌 공유 (decode용 Q4M1 사본 255 MiB 제거)

### 2) 첫 토큰 (graph init + KV seed)

- 기존 : 첫 토큰 3.0~3.2 s, 그중 KV seed(prefill KV → decode int8 cache) 2.3~2.5 s

| 변경 | p1024 | p2048 |
|---|---|---|
| seed를 DSP worker pool / CPU pool로 (S25U) | 3.29 → 1.53 s | 5.66 → 2.69 s |
| seed abs-max, append 벡터화 | 1,463 → 1,013 ms | 2,206 → 1,411 ms |
| fp16 그대로 넘기고 DSP에서 넓힘 | host 변환 145 ms → 0 | 288 ms → 0 |
| RoPE 테이블 30번 복사+비교 → 이름으로 공유 | → 506 ms | → 636 ms |

### 3) Attention (decode)

| 변경 | 효과 |
|---|---|
| KV cache int8, scale은 (층, kv head, dim)별 고정 (S25U) | ATTN_M1 20.9 → 11.4 ms/token, cache 440 → 220 MiB |
| PV에서 V 행을 64폭마다 다시 읽던 것 → 한 번에 (S25U) | 11.5 → 10.4 ms |
| integer attention (int16 q, int32 score/PV) | 10.36 → 6.73 ms/token |
| sliding 층은 window(1024행)만 KV 유지 | decode cache 640 → 106 MiB, 토큰당 143 → 116 ms (6,065토큰) |

### 4) expert cache

- C=16 상주, 나머지는 miss → miss 대기가 decode 시간의 큰 부분 (TL2 p2048 22.6 ms/token)

| 변경 | decode |
|---|---|
| LRU → LRFU (32토큰 반감기), C=32 | p1024 10.83 → 11.93 TPS, p2048 10.44 → 12.16 TPS |
| decode seed 후 CPU K/V 사본 해제 (S25U) | 11.97 → 12.27 TPS, MemAvailable +470~520 MiB |
| 첫 토큰 후 expert pool 확장 (prefill staging, KV가 비운 PD 공간 → 층당 34개까지) | C16 10.40 → 13.07 TPS (arena 2,240 MiB) |
| 다음 layer expert 예측 + DMA queue 4개 등 빌트인 | 9.6~10.0 → 10.3~10.6 TPS |
| TL2 gate/up + down read 1번으로 | 11.96 → 12.57 TPS, expert 대기 24.74 → 22.62 ms/token |
| demand-read 전용 pool, router weight BF16, 불필요 logits 복사 제거 | 12.47 → 13.26 TPS |

다음 layer expert 예측 prefetch
- DSP가 layer L 도중에 router 입력 행을 mailbox에 올림
- ARM이 layer L+1 router weight로 top-k 예측 (직전 토큰의 layer 간 residual 변화량 0.5 blend)
- pool에 없는 expert만 reader 4개가 여분 DMA slot 32개에 미리 읽음
- 실제 miss 시 등록 RPC 없이 slot만 교환, 실제 routing은 항상 DSP 결과
- 예측만 따로 측정한 수치는 없음 / DSP 쪽 예측 + page cache 데우기 방식은 오히려 느려서(11.4 → 10.6 TPS) 기본 off

### 5) lm_head

- untied head를 NPU에서 64개 WH slice로 (slice 4096열, VTCM 6 MiB 제한)
- prefill 마지막 행과 decode가 같은 slice 사용 → CPU용 head 사본 396 MiB 제거
- decode 13.09 → 13.41 TPS, head load 852 → 124 ms (slice 캐시 파일)

decode 결과 (S26U)

| 조건 | decode |
|---|---|
| QS2 pack, 1,028토큰 요약 (10/13) | 12.3 TPS |
| QS2 pack, 같은 프롬프트 (10/14) | 13.8 TPS |
| TL2 pack, 1024토큰 C=16 (10/18) | 12.32 TPS |
| rotated pack, 30층 전부 TL2 (10/19) | 15.5 TPS |
| strict rotated pack, 2K 요약 (10/20) | 19.54 TPS |
| 30k context (S25U, C=8) | 6.58~6.69 TPS (ATTN_M1이 51%) |

---

## 4. 메모리

```
DDR
├─ ARM 프로세스 힙 ─────────────── tokenizer, embedding 행 버퍼
│                                   prefill 중 host KV(fp16) → decode 시작 때 해제
│
├─ ION staging (캐시 있음) ──────── 층 사이 activation / 블록 안에서는 DSP에 그대로 유지
│
├─ ION arena (캐시 없음) ────────── expert pool slot (C=16 → 첫 토큰 후 최대 34)
│                                   + FC / dense FFN WH weight + lm_head slice
│                                   C16 1,728 MiB / pool 확장 후 2,240~2,368 MiB
│
├─ rpcmem KV cache ──────────────── int8 decode KV (sliding 층은 window만)
│
└─ DSP 힙 ───────────────────────── router weight, norm gamma, RoPE table, 그래프 op table 등

PD 매핑 한도 3,840 MiB (여유 256 MiB) 안에 arena + DSP 힙 + KV가 다 들어가야 함

DSP 칩 안
└─ VTCM 8 MiB ──────────────────── HMX 타일, decode expert weight 2 slab double buffer,
                                     lm_head slice는 6 MiB 이하로
```

| 항목 | before | after |
|---|---|---|
| 30k context KV cache 할당 | 6.5 GB (908,160행) | 495 MB (69,120행) |
| sliding 층 decode cache (6,065토큰) | 640 MiB | 106 MiB |
| decode KV cache int8 | 440 MiB | 220 MiB |
| page cache 제한 (2048토큰) | peak 8.4 GB | 3.6 GB (decode 9.9 → 7.0 TPS 손해) |
| expert 30층 전부 TL2 | arena 2,624 MiB / peak 4,723 MB | arena 1,792 MiB / peak 3,669 MB |
| TL2 expert slot (padding 제거) | 1,523,712 B | 1,290,240 B |
| lm_head NPU | CPU 사본 396 MiB | 0 |
| FC weight 공유 | Q4M1 사본 255 MiB | 0 |
| RoPE table | 262,144 위치 전부 (fp32 1.5 GiB + fp16 768 MiB, OOM) | 실제 길이만큼 |

peak (host anon + DSP) : 1,028토큰 요약 4.2 GB (10/13)

### expert 1개 크기

expert 1개 = gate_up [2816 → 1408] + down [704 → 2816] = weight 5,947,392개

| 포맷 | gate_up | down | slot (scale, column sum 포함) |
|---|---|---|---|
| BF16 원본 | 7.56 MiB | 3.78 MiB | 11.3 MiB |
| int4 WH (QS4CX_WH, 28/29층 affine) | 1.89 MiB | 0.95 MiB | 3.0 MB |
| 2-bit QS2CX_WH | 0.95 MiB | 0.47 MiB | 1,523,712 B (1.45 MiB) |
| ternary TL2CX_WH | 811 KiB | 413 KiB | 1,290,240 B (1.23 MiB) |

- 참고 : LFM2 expert 1개 = 5.25 MiB (int4, gate_up 3.5 + down 1.75)
- decode 토큰당 expert 240개 (8개 × 30층) 읽음 → QS2 약 349 MiB, TL2 약 295 MiB (계산값)
- slot 크기와 3.0 MB는 커밋 기록값, 나머지는 shape으로 계산한 값

### VTCM 구성

VTCM 하나(8 MiB)를 콜마다 용도에 맞게 다시 나눠 씀. 배치는 kernel이 콜 시작 때 shape으로 계산함
(`hexkl_mm_u8i4_moe_layout`). 아래 크기는 Gemma4 shape을 넣어 계산한 값

prefill MoE 콜 (64행 블록 단위)

```
VTCM 8 MiB
├─ activation 64행 블록 ──── 176 KiB   64 × 2816 u8 (HMX 타일 순서)
├─ gate_up weight ────────── 1.89 MiB  int4로 펼친 상태
├─ down weight ───────────── 0.95 MiB
├─ gate 버퍼 ─────────────── 176 KiB   64 × 704 f32, silu(gate)·up 결과 (up은 따로 안 만듦)
├─ mid 버퍼 × 2 ──────────── 88 KiB    down 입력 u8, 블록 n / n+1 교대
├─ acc staging × 2 ───────── 512 KiB   32 tile × 8 KiB, HMX가 하나 채우는 동안 HVX가 다른 하나 dequant
├─ down staging ring × 4 ─── 1 MiB
│                             ── 합계 약 4.77 MiB
├─ Hadamard HMX 모드일 때 ── 328 KiB
└─ HMX config (맨 위)
```

- weight는 2벌 안 둠 → 시간차 재사용
  - expert e의 down 계산 중 → 이미 다 쓴 gate_up 영역에 e+1의 gate_up DMA
  - 이유 : LFM2 shape에서 2벌이면 10.5 MB > VTCM arena 8.3 MB (Gemma4 shape이면 2벌도 계산상 들어가지만 같은 kernel이라 구조는 동일)
- weight는 32 column tile 단위 chunk로 DMA → 첫 chunk 도착하면 HMX 시작 (계산상 gate_up 2 chunk, down 3 chunk)
- 2-bit QS2 : 2-bit 코드를 int4 자리의 위쪽 절반에 DMA → 그 자리에서 int4로 펼침 (추가 VTCM 없음, DMA 바이트 절반)
- TL2 : native panel을 DMA → int4 WH로 펼침 (VTCM 여유 있으면 panel을 연속으로 받아 둠)

decode MoE 콜 (M=1, VTCM feed)

```
VTCM 8 MiB
├─ slab G0 ───────── 968 KiB   expert 0, 2, 4, 6의 gate_up
├─ slab G1 ───────── 968 KiB   expert 1, 3, 5, 7의 gate_up
│                     gate_up 다 쓰면 slab 하나에 down 2개씩 (down j → slab (j>>1)&1, 반쪽 j&1)
├─ TL2 LUT ───────── gate_up 59 KiB + down 15 KiB × 8
└─ 나머지 미사용
```

- slab 크기 = int4 gate_up의 절반 (2-bit / TL2는 packed 그대로 올리고 int4로 안 펼침)
  - TL2 실제 크기 gate_up 811 KiB, down 413 KiB → slab 안에 들어감
  - 28/29층 affine 4-bit expert면 slab 1.89 MiB
- expert i 계산하는 동안 다음 expert를 DMA → double buffering
- DMA는 queue 4개로 나눠 보냄 (queue 1개 28~34 GB/s → 4개 62 GB/s)

그 밖의 콜

| 콜 | VTCM 사용 |
|---|---|
| decode FC (qkv, o proj, dense FFN) | lane마다 VTCM을 나눠 갖고, 각자 block 4개를 두 버퍼로 교대 (block b 계산 중 b+1 DMA) |
| lm_head | slice 1개 ≤ 6 MiB (4096열 × 2816 int4 = 5.5 MiB), 64 slice |
| prefill attention | head 하나의 K^T / V 타일 상주, score는 VTCM 안에서 softmax까지, Q는 다음 블록 미리 DMA |
| prefill FC | weight slice (K ≥ 4096이면 1 MiB) + VTCM에 맞는 행 수만큼씩 나눠서 처리 |

---

## 5. weight 포맷

| 포맷 | 대상 | 비고 |
|---|---|---|
| QS2CX_WH | expert (초기) | 2-bit 코드 + int4 4-level palette, decode는 2-bit 그대로 LUT GEMV |
| TL2CX_WH | expert | ternary (-1/0/1), 약 1.67 bit |
| QS4CX(_WH) | projection, dense FFN | int4 + 열별 scale, WH = HMX 타일 순서로 미리 배치 |
| QS4CX_WH_AFFINE | FC, untied head | int4 + scale + offset |
| Q4_0 | embedding | CPU에서 필요한 행만 |
| int8 KV | attention cache | (kv head, dim)별 고정 scale |

- WH 포맷을 변환기에서 미리 만들어 둠 (load 때 변환하면 157 MB/s → 25초)
- rotated checkpoint : expert 입력에 online Hadamard 회전 (HVX FP32 decode / HMX FP16 prefill)
- 30층 전부 TL2 vs 28/29층 4-bit affine (같은 ternary 값)
  - prefill 577 → 613 TPS, decode 11.8 → 15.5 TPS, peak 4,723 → 3,669 MB

---

## 6. 최종 결과

| | prefill 1K | prefill 2K | decode | peak |
|---|---|---|---|---|
| 처음 (10/10) | 402 TPS | 398 TPS | 10~13 TPS | |
| QS2 pack (10/13~14) | 627~630 TPS | 693 TPS | 13.8 TPS | 4.2 GB (rebase 전 head 5.1 GB) |
| TL2 pack (10/15~18) | 634~650 TPS | 699~700 TPS | 12.3~13.3 TPS | |
| strict rotated pack (10/20) | | 683.70 TPS | 19.54 TPS | |

참고 : 같은 SoC GPU (Adreno 840, OpenCL, ternary TQ2CX expert, pack 다름)

| | prefill 1K | decode | peak RSS |
|---|---|---|---|
| GPU | 388 TPS | 25.3 TPS (resident) / 20.3 TPS (streaming) | 7.8 GB / 3.4 GB |
| NPU | 627~630 TPS | 13~15 TPS (QS2/TL2) / 19.54 TPS (strict) | 3.6~4.2 GB |

→ prefill과 메모리는 NPU가 유리, decode는 GPU가 더 빠르거나 비슷

---

## 추후 계획

- decode 16 TPS 목표 (QS2/TL2 pack 미달, strict pack 19.54)
- static activation 모드 속도 (현재 prefill 82.16 / decode 16.86 TPS, dynamic 677.82 / 19.42 TPS)
- PetCare sub-agent TC 품질 (기대 `turnOnPetMonitoring` → 실제 `launchPetMonitoring`)
- 긴 context decode attention (30k에서 ATTN_M1 51%, 실효 대역폭 약 4.2 GB/s)
- max_seq_len에 비례해 load 때 잡히는 메모리 줄이기 (30,272에서 peak 5.25 GB)
- QuickDotAI API에 KV snapshot / resume 노출 (현재 native CLI만)
