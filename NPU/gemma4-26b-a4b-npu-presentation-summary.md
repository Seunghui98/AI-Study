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
  - weight 6.2~7.0 GiB, NPU 매핑 한도 3.75 GiB (3,840 MiB)
  - expert는 layer당 16개만 상주(C=16), 나머지는 flash에서 읽음 (decode 중에는 첫 토큰 후 32개까지 늘림)
- expert 수 : LFM2 대비 4배 (layer당 32 → 128개, top-8, 30 layer)
- NPU 연산 증가 → FastRPC 콜 / 복사 비용이 matmul보다 커짐

---

## 2. Expert weight 포맷 : 2-bit → ternary

- expert weight를 2 bit에서 ternary(-1/0/+1) packing, weight 1개당 평균 1.67 bit로
- decode는 expert weight 읽기가 병목(memory bound) → 읽는 양이 줄어 성능 개선

QS2 pack (2 bit)
- weight 하나 = 2 bit 번호 (00/01/10/11) → 변환표(tensor마다 int4 값 4개)로 값을 읽음
- 원본 google checkpoint를 4개 값으로 양자화해서 만든 pack

TL2 pack (ternary)
- weight 하나 = -1 / 0 / +1 셋 중 하나 → 변환표 없이 커널이 값을 바로 읽음
- 2 bit는 칸이 4개인데 값은 3개라 1칸이 늘 빔 → weight 3개를 묶어 5 bit에 (27가지 ≤ 32칸) = 1개당 1.67 bit
- 처음부터 ternary로 양자화된 checkpoint(AIMET)를 pack한 것 → QS2 pack과는 checkpoint도 다름

| | QS2 pack | TL2 pack |
|---|---|---|
| model size | 7.02 GiB | 6.15 GiB |
| expert 1개 | 1.45 MiB | 1.23 MiB |
| decode 토큰당 읽는 expert weight (240개) | 349 MiB | 295 MiB |
| FC / lm_head | int4 | int4 + offset (BF16 원본 값 복원) |

```
prefill : packed ──DMA──▶ VTCM ──HVX가 int4로 펼침──▶ HMX
decode  : packed 그대로 ──DMA──▶ VTCM ──▶ HVX GEMV (표 조회 + 부호 뒤집기)
```

- 효과 (28/29층 4-bit 컨테이너 → 30층 전부 ternary) : decode 11.8 → 15.5 TPS, arena 2,624 → 1,792 MiB, peak 4,723 → 3,669 MB
- 같이 들어간 것 : HMX 타일 순서(WH)로 변환기가 미리 배치, rotated checkpoint(Hadamard 회전), 변환기 1개로 HF → pack

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

첫 토큰 준비 : KV cache seed
- 구조 : prefill과 decode는 attention 커널이 달라서 KV cache도 따로 (레이아웃, scale 단위가 다름)
- seed = prompt의 K/V(host fp16)를 decode용 int8 cache로 옮겨 담는 작업, 층마다 1번
- 처음엔 첫 토큰 3.0 s 중 seed가 2.3 s

```
층 1개의 seed
host : fp16 K/V 행 ──전송──▶ DSP : (head, dim)별 |max| → scale → int8 양자화 → decode cache에 append
```

- 줄인 것

| 방법 | 효과 (p1024) |
|---|---|
| 절댓값 최대 / 양자화를 DSP 스레드 1개 → worker pool 8 스레드(7 worker + 호출 스레드) + HVX 벡터화 | seed RPC 429 → 93 ms |
| host에서 fp16→f32 변환 안 하고 그대로 전송, DSP에서 넓힘 | host 변환 145 → 0 ms |
| RoPE 테이블 층마다 30번 전송 → 1번 공유 (그래프 초기화 쪽) | 439 ms 제거 |

- 첫 토큰 1,463 → 506 ms (p2048 : 2,206 → 636 ms)
- seed 끝나면 host fp16 사본 해제 → 470~520 MiB를 expert cache가 씀

Attention
- KV cache int8 + 정수 연산 attention
- 토큰당 attention 20.9 → 6.7 ms (S25U → S26U)

expert cache
- 구조 : 층마다 expert 128개 중 16개만 DSP 메모리에 상주 (설정 C=16), 나머지는 flash
  - 첫 토큰이 나오면 prefill 버퍼가 비운 자리만큼 자동으로 늘림 → decode 중에는 층당 32개 (한도 34)
- 토큰마다 층당 8개 선택 → 상주에 없으면 miss → flash에서 읽을 때까지 대기
- decode 시간의 대부분이 이 miss 대기 (TL2 p2048 : 토큰당 22.6 ms)

```
토큰 1개의 한 층
DSP  : router → 8개 중 5개 있음 → 5개 먼저 계산 ─────────────▶ 나머지 3개 계산
ARM  :                  miss 3개 받음 → flash에서 읽어 빈 slot에 ─┘
```

- 줄인 것

| 방법 | 효과 |
|---|---|
| 상주 수 늘리기 : 첫 토큰 후 16 → 32개 (+16, arena 2,240 MiB) | 10.4 → 13.1 TPS |
| 교체 정책 LRU → LRFU (최근 + 빈도) | 10.8 → 11.9 TPS (C=32) |

```
교체 정책 (꽉 찬 cache에서 누구를 내보낼지)
LRU  : 가장 오래 안 쓴 expert부터 내보냄 → 자주 쓰던 expert도 잠깐 안 쓰면 바로 밀려남
LRFU : expert마다 "사용 횟수"를 두고 32토큰마다 반으로 줄임 → 점수 가장 낮은 것부터 내보냄
       (최근에 자주 쓴 expert는 점수 높음 = 오래 남음, 동점이면 오래 안 쓴 쪽)
       시뮬레이션 : miss 2.79 → 2.23 개/콜 (C=24)
```

| expert 크기 줄이기 : 2-bit → ternary | 11.8 → 15.5 TPS |
| 다음 층 expert 예측해서 미리 읽기 | 보조 (단독 수치 없음) |

- 결론 : 커널보다 "몇 개를 올려 둘 수 있나"가 decode 속도를 정함 → 메모리 절감이 곧 decode 성능

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

## 7. Long context (agent 시나리오)

- 시나리오 : 바뀌지 않는 긴 앞부분(system prompt, tool 선언) + 짧은 요청
  - sub agent 1,537 + 16 토큰 / main agent 23,063 + 6,641 토큰 (context 30k)
- 문제
  - KV cache를 층마다 max_seq_len으로 잡으면 30k에서 6.5 GB
  - full attention 층이 4,096열을 넘으면 느린 fp16 경로로 떨어짐
  - 긴 앞부분을 요청마다 다시 prefill

처리

| 방법 | 내용 | 효과 |
|---|---|---|
| 층별 KV 크기 | sliding 25층은 window 1024 + chunk 2048 = 3,072행만, full 5층만 전체 길이 | 30k KV 6.5 GB → 495 MB |
| chunk prefill | 2,048행씩 나눠 prefill, chunk 사이에 sliding window를 앞으로 당김(compaction) | 20k 이상 prefill 가능 |
| attention 커널 | 4,096열 넘으면 segment로 나눠 계산 후 softmax merge | 22k+6k prefill 69.5 → 39.9 s |
| KV snapshot | 앞부분을 한 번 prefill해 파일로 저장(.kv + scale .q2) → 실행 때 load 후 뒷부분만 prefill | load 200 ms (6k) / 292~684 ms (23k) |
| window cache 정확도 | decode scale이 seed 행 기준이라 window만 seed하면 full cache와 달라짐 → 버린 행의 최댓값을 같이 넘김 | max_seq_len과 무관하게 같은 출력 |
| 입력 검사 | prefix + 프롬프트 + 생성 길이 > max_seq_len이면 prefill 전에 거부 | RoPE 범위 초과로 죽던 것 방지 |

왜 2,048행 chunk로 나누나
- 메모리 : prefill 버퍼(activation, staging, DSP scratch)가 한 번에 넣는 행 수에 비례
  - 예 : qkv 출력 버퍼 2,048행에서 128 MiB, MoE scratch도 행 수 비례
  - 행을 늘리면 DSP 매핑 한도(3,840 MiB)를 expert cache와 나눠 써야 해서 OOM 또는 C를 낮춰야 함 (4,096행 한 번에 : C=24로 낮춰야 돎)
- KV cache : sliding 층 cache = window 1,024 + chunk 1개 2,048 = 3,072행 → chunk 크기가 곧 sliding cache 크기
- 속도 : 2,048행이면 NPU가 이미 포화 (1K 627 → 2K 700 TPS, 더 키워도 이득 작음)
→ chunk 크기를 고정하면 prompt 길이와 무관하게 prefill 메모리가 일정

결과

| 조건 | prefill | decode |
|---|---|---|
| S26U, 19,270토큰 한 번에 (C=8) | 385~394 TPS | 7.3~7.9 TPS |
| S26U, 22k cached + 6k 새 토큰 | 39.9 s (152 TPS) | 토큰당 155 ms |
| S25U, 23k 저장 → 6,641 prefill (chunk 4개) | 21.9~25.0 s (266~304 TPS) | 6.6 TPS |
| 참고 2k | 700 TPS | 13~15 TPS |

- 긴 context에서 느려지는 이유 : full attention 5층이 context 전체를 봄 (30k prefill 25 s 중 full 층 attention 7.5 s, decode는 attention이 51%)
- 남은 것
  - full 층 KV는 아직 max_seq_len에 비례해 load 때 할당 (30,272에서 peak 5.25 GB)
  - prefill 중 host fp16 KV 사본 약 1.2 GB (decode 시작 때 해제)
  - 30k에서는 expert cache C=8, pool 확장 off 필요 (DSP 매핑 공간을 KV가 씀)

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
