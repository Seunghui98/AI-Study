# Gemma4 26B-A4B NPU(Hexagon HTP) 최적화 정리 — nntrainer PR #4428

대상 PR: [nntrainer/nntrainer#4428](https://github.com/nntrainer/nntrainer/pull/4428)
("[CausalLM] Enable Gemma4 MoE HTP in QuickDotAI and SampleTestAPP")

이 문서는 PR의 커밋 메시지(626개)와 PR 안의 문서(`docs/gemma4-htp-*.md`)에 기록된
측정값을 모아서 정리한 것이다. 숫자는 전부 커밋/문서에 적힌 값을 그대로 옮겼고,
새로 측정한 것은 없다. tok/s로 환산한 값은 "환산"이라고 표시했다.

---

## 0. 요약

Gemma4 26B-A4B는 전체 26B, 토큰당 활성 4B인 MoE 모델이다(층 30개, 층마다 expert
128개 중 8개 선택). 이걸 Galaxy S26 Ultra(Snapdragon SM8850, Hexagon v81)의 NPU에서
prefill과 decode를 모두 돌리도록 만든 것이 이 PR이다.

| 항목 | 처음 (S26 Ultra, 10/10 전후) | 최종 (10/14 ~ 10/20) |
|---|---|---|
| Prefill 1024토큰 | 2,549 / 2,716 ms (환산 377~402 tok/s) | 1,632 ms (환산 627 tok/s), 1,028토큰 1,633 ms (630 tok/s) |
| Prefill 2048토큰 | 398 tok/s | 699~700 tok/s, 2K 한국어 요약 683.70 tok/s |
| 첫 토큰 setup (p1024) | 3.0~3.2 s → 1,463 ms | 506 ms |
| Decode | 10~13 tok/s (QS2 pack, 빌드 기본값 9.6~10.0) | 13.1~13.8 tok/s (QS2 pack), 15.5 tok/s (all-TL2 pack), 19.54 tok/s (strict rotated pack) |
| Peak 메모리 (host anon + DSP) | 5.1 GB (10/13 rebase 전 head) | 4.2 GB (10/13), all-TL2 pack 3,669 MB |
| 30k context KV cache 할당 | 6.5 GB | 495 MB |

크게 보면 prefill은 "FastRPC 호출과 메모리 복사를 줄이고 HMX를 쉬지 않게 하는" 방향,
decode는 "토큰 하나를 DSP 안에서 끝내고, expert 가중치 읽기(대역폭)를 줄이는" 방향,
메모리는 "같은 데이터를 두 벌 들고 있지 않게 하는" 방향으로 정리된다.

Decode는 같은 weight pack 안에서는 많이 오르지 않았다(약 10 → 13 tok/s). decode는
가중치 대역폭이 거의 전부라서, 큰 폭의 개선은 커널보다 weight 포맷(2-bit → ternary
TL2)과 expert cache 크기에서 나왔다.

---

## 1. 배경: 왜 NPU에서 돌리기 어려운가

### 1.1 하드웨어에서 신경 써야 하는 것

| 구성 | 설명 | 이 작업에서 의미 |
|---|---|---|
| HMX | 행렬 곱 유닛(u8 x i4, u8 x i8) | prefill matmul 담당. 데이터가 VTCM에 타일 형태로 있어야 함 |
| HVX | 1024-bit 벡터 유닛, 스레드 여러 개 | 양자화, norm, softmax, epilogue, decode GEMV |
| VTCM | DSP 로컬 scratch, 약 8 MB | 가중치/활성 타일을 DMA로 올려서 씀. 여기에 안 맞으면 쪼개야 함 |
| DMA | DDR → VTCM 전송 | expert 가중치 공급 속도를 좌우 |
| FastRPC | ARM ↔ DSP 호출 | 호출마다 고정 비용 + 버퍼 크기에 비례하는 cache 유지 비용(MiB당 약 27 us) |
| rpcmem(ION) | ARM과 DSP가 같이 보는 메모리 | weight arena, expert cache, KV cache, 층 사이 활성 |
| PD 매핑 한도 | DSP 프로세스가 매핑할 수 있는 총량 3,840 MiB | expert cache, KV, FC 가중치를 전부 이 안에 넣어야 함 |

### 1.2 모델 쪽 문제

- 가중치가 크다. QS2 pack 기준 7.5 GB, TL2 pack 6.6 GB. 3,840 MiB 안에 다 못 올린다.
  → expert는 층당 16개만 상주시키고 나머지는 파일에서 필요할 때 읽는다(expert streaming).
- 층마다 연산이 잘게 나뉘어 있다(RMSNorm 9개, q/k/v/o, router, dense FFN, MoE, residual).
  층 단위로 하나씩 FastRPC를 부르면 호출 비용과 복사가 matmul보다 커진다.
- vocab이 262,144로 크다. lm_head만 352 MiB(WH 기준)라 VTCM 크기에 맞게 64조각으로 나눠야 한다.
- sliding window 층 25개 + full attention 층 5개 구조라서, 긴 context에서 KV cache를
  층마다 max_seq_len만큼 잡으면 메모리가 터진다.

---

## 2. 전체 구조

### 2.1 prefill과 decode는 완전히 다른 경로

**Prefill**은 nntrainer 그래프를 층 단위로 돈다. HTP 층마다 입력 행을 rpcmem에 올리고
skel 함수 하나(qkv, attention, MoE 등)를 호출한다. 그래서 prefill 최적화의 대부분은
"호출 수 줄이기, 호출 사이 복사 없애기, 한 호출 안에서 HMX/HVX/DMA 겹치기"다.

**Decode**는 첫 토큰이 나온 뒤에는 층을 거치지 않는다. 한 토큰 step 전체(542개 op:
RMSNORM, FC, QK_NORM, ROPE, ATTN_M1, ADD, ROUTER_TOPK, MOE, DENSE_FFN, LM_HEAD)를
DSP 안에 상주 그래프로 만들어 두고(`graph_init`), token driver가 돈다. ARM은 위치를
mailbox에 넣고, DSP는 그래프를 돌려 argmax token id만 돌려준다. 토큰당 메시지 하나이고
그 사이 CPU는 논다.

### 2.2 가중치 포맷

| 포맷 | 대상 | 특징 |
|---|---|---|
| `QS2CX_WH` | routed expert 128개 (초기 pack) | 2-bit 코드 + int4 4-level palette. decode는 2-bit 그대로 LUT GEMV, prefill은 VTCM에서 int4로 펼침 |
| `TL2CX_WH` | routed expert (TL2 pack) | ternary(-1/0/1), 약 1.67 bit. decode는 TL2 LUT GEMV, prefill은 int4로 펼쳐서 HMX |
| `QS4CX` / `QS4CX_WH` | attention projection, dense FFN | int4 + 열별 fp32 scale. `_WH`는 HMX 타일 순서로 미리 배치해 둔 것 |
| `QS4CX_WH_AFFINE` | FC, untied head, 일부 expert 층 | int4 + scale + offset (`w = scale * (q + offset)`) |
| `Q4_0` | embedding (+ tied head) | GGML 포맷, CPU에서 필요한 행만 읽음 |
| int8 KV | attention cache | (kv head, dim)별 고정 scale |

WH 포맷을 따로 둔 이유: load 때 CPU에서 HMX 레이아웃으로 바꾸면 약 157 MB/s라 모델
하나에 25초가 걸린다. 변환기가 처음부터 HMX 타일 순서로 써 두면 load는 그냥 읽기만 한다.

### 2.3 최종 빌트인 런타임 설정 (`nntrainer/tensor/htp_fixed_opts.h`)

실험하면서 찾은 값들을 환경 변수 대신 상수로 박아 넣었다. 이전에는 환경 변수를 안 주면
decode가 9.8 tok/s, 주면 13.9 tok/s여서 CLI가 느려지는 문제가 있었다.

| 설정 | 값 | 의미 |
|---|---|---|
| E2E | on | decode를 상주 그래프 + token driver로 |
| MOE_CACHE_EXPERTS | 16 | 층당 상주 expert 수 (C=16) |
| ATTN_M1_Q8 | 2 | decode attention을 int8 KV로 |
| FC_IN_ARENA | on | dense projection을 DSP arena에 |
| E2E_FREE_KVQ / KV_DROP_CPU / DROP_HOST_FC | on | DSP가 들고 나면 host 사본 해제 |
| MOE_DMA_QUEUES | 4 | M=1 expert 공급을 DMA queue 4개로 분할 |
| PREDICT | 2 | ARM이 다음 MoE 층 expert를 router 입력으로 예측 |
| PREFETCH_READERS / SLOTS | 4 / 32 | 예측한 expert를 여분 slot에 미리 읽기 |
| RELEASE_PREFILL_SCRATCH | on | 첫 토큰 후 prefill scratch 해제 |
| MOE_POOL_TARGET | 34 | 첫 토큰 후 PD 여유만큼 expert pool을 층당 34개까지 키움 |

---

## 3. Gemma4 이전에 깔린 공용 기반 (LFM2 / Qwen3에서 먼저 만든 것)

PR 히스토리의 앞쪽 절반(8월 ~ 9월 말)은 Gemma4가 아니라 LFM2.5-8B-A1B(MoE)와
Qwen3-0.6B로 HTP 백엔드를 처음 올린 작업이다. 장치도 주로 S25 Ultra(v79)다. Gemma4는 이
위에 얹혔기 때문에, 설명할 때 "기반"으로 따로 묶어 두는 게 이해하기 쉽다.

| 최적화 | 내용 | 효과 (LFM2 기준) |
|---|---|---|
| 층 하나 = MoE 호출 하나 | expert마다 부르던 호출을 층 단위 batched kernel로 | transport 16.95 → 2.65 ms, 층 154.0 → 63.5 ms |
| prefill MoE 파이프라인 | HMX batch 하나가 도는 동안 앞 batch의 HVX epilogue(dequant/SwiGLU/scatter)를 worker pool에서 | quant 1440 → 354 us/call, scatter 1056 → 23 us |
| VTCM feed (decode) | expert 가중치를 HVX가 ION arena에서 직접 읽던 것을 DMA로 VTCM에 한 expert 앞서 올려 둠 | M=1 mm 922.7 → 676.6 us (-27%), decode 28.22 → 37.38 tok/s (+32.5%) |
| DMA L2 bypass | expert 가중치 DMA descriptor에 src_bypass | DMA 37.3 → 69.3 GB/s, decode +31~35% |
| Expert streaming | expert를 파일 offset만 들고 있다가 필요할 때 arena slot으로 읽음. 여러 swap을 한 RPC로 묶고, slot을 제자리에서 rebind | cold prefill 2318 → 935 ms (read-ahead 기본 on) |
| ION arena | DSP heap(1.89 GB 한계) 대신 rpcmem arena를 DSP에 매핑 | DMA 29.96 GB/s 유지하면서 용량 한계 해소 |
| load 시 weight 등록 | transpose 타일화, 병렬 bake, 오프라인 WH 변환 | 등록 3819.6 → 1674.7 ms |

DMA bypass, VTCM feed, expert streaming, 상주 decode 그래프, 토큰당 FastRPC 하나
구조는 이때 만들어진 것을 Gemma4가 그대로 쓴다.

---

## 4. Prefill 최적화

### 4.1 흐름

Gemma4 prefill을 처음 NPU에 올렸을 때(10/02, 447토큰 프로파일)는 MoE만 HTP였고,
5,389 ms 중 q/k/v/o projection 1,540 ms, dense MLP 802 ms가 CPU에 남아 있었다(44%).
그래서 순서는 대략 이렇게 갔다.

1. 나머지 층을 전부 DSP로 옮기기 (projection, dense FFN, router, RoPE, norm)
2. 옮기고 나니 생긴 호출 비용/복사/staging을 줄이기 (fusion)
3. 커널 안에서 쉬는 유닛 없애기 (MoE, attention, QKV epilogue)
4. weight pack이 TL2로 바뀌면서 생긴 느린 경로 다시 벡터화

### 4.2 층을 DSP로 옮기고 호출을 합침

- **projection/dense FFN을 HTP로**: q/k/v를 `qkv_layer` 하나로, dense MLP를 `dense_ffn` 하나로
  합쳤다. GeGLU는 MoE 커널 epilogue에서 처리한다. 층당 HTP 호출이 7개에서 2개로 줄었다.
- **RMSNorm 9개 중 7개를 호출 안으로**: norm은 행 단위 연산이라, 이미 행을 들고 있는 matmul
  호출 안에서 하면 된다. 이전에는 DSP 호출 사이마다 CPU가 activation 전체를 한 번 더
  훑었다(512토큰 prefill에서 약 300 ms).
- **epilogue fusion**: o-proj 다음 residual_add, MoE 다음 post-FFN norm+add를 각각 같은
  호출 안으로 넣었다. 호출 30개씩 줄었고 staging 트래픽이 7.2 → 6.2 GB,
  5.56 → 4.9 GB로 줄었다(S25 Ultra, 1024토큰).
- **router를 DSP로**: router logits, norm, softmax, top-k까지 한 호출. gate 가중치는 VTCM에서
  읽고, top-k는 128개를 13번 훑던 것을 정렬 삽입 한 번으로 바꿨다. router 행 267 → 213 ms.
- **RoPE를 qkv 호출 안으로**, RoPE 테이블은 max_position_embeddings(262,144)가 아니라
  실제 쓸 길이(max_timestep)만큼만 만든다. 이전에는 fp32 1.5 GiB + fp16 768 MiB를 잡아서
  S25가 첫 decode 토큰에서 OOM으로 죽었다.

이 단계(10/08, S25 Ultra, 1024토큰) prefill 시간 변화:

| 커밋 | 변경 | prefill (ms) |
|---|---|---|
| 6839eae | host 쪽 Q range/변환 루프 NEON화 (calibration 398 → 39 ms) | 4,791~4,796 → 4,605~4,661 |
| 98869e9 | 양자화 KV cache를 첫 prefill이 아니라 load 때 등록 | → 4,420 / 4,771 / 4,483 |
| 523c840 | KV append 양자화를 HMX 스레드 혼자가 아니라 HVX worker들이 | → 4,217 / 4,260 / 4,453 |
| cacc9a8 | HMX 곱을 라이브러리 호출(50 cycle) 대신 inline으로 (실제 HMX는 9 cycle) | → 4,048 / 4,074 / 3,995 |
| 3d7661f | router top-k 한 번에 | → 4,205 / 3,935 / 3,924 |
| 02d899e | MoE에 post-FFN epilogue 포함 | → 3,911 / 4,175 / 3,880 |
| a2846b7 | FC 활성 양자화의 파라미터 계산과 pack을 한 번에 (행을 DDR에서 두 번 읽지 않음) | → 3,866 / 3,918 / 3,839 |
| f8d8cb7 | o-proj를 residual_add 호출 안으로 | → 3,808 / 3,822 / 3,813 |

참고로 HMX router(u8 x i8, 7118758)는 router 행 214 → 99 ms로 빨랐지만 logits가 CPU와
0.17~0.39 달라 routing 자체가 바뀌어서 기본은 꺼 두었다.

### 4.3 staging: 블록 안 텐서는 DSP에 그대로 둔다 (6bd991e)

한 decoder 블록의 텐서들이 호출 사이에 host로 돌아왔다가 다시 ION으로 복사되고 있었다.
이제 블록 안에서는 DSP staging에 그대로 두고, host에는 확인용 샘플 2 KB만 넘긴다.

| | before | after |
|---|---|---|
| staging memcpy p1024 | 143~151 ms (2,940 MB) | 22 ms (301 MB) |
| staging memcpy p2048 | 402 ms (7,444 MB) | 39 ms (408 MB) |
| prefill p2048 | 4,728 / 5,020 / 5,293 ms | 4,438 / 4,178 / 4,438 ms |

이어서 staging 버퍼를 2의 거듭제곱 크기가 아니라 실제 텐서 크기로 잡고(FastRPC cache
유지가 dma-buf 전체에 걸리기 때문에 11.5 MB 텐서가 16 MiB 비용을 냈다), 이미 다 읽힌
텐서는 host로 복사하지 않게 했다(1024토큰에서 87번, 957 MB 복사 제거).

### 4.4 MoE prefill 커널

| 커밋 | 내용 | 효과 |
|---|---|---|
| 4b6d2ad | 행 하나가 expert 8개로 가면 8번 양자화하던 것을 한 번만. down scatter는 행 순서로 | worker 시간: down epilogue 128 → 74 ms, pack 63 → 27 ms. prefill p1024 2,259~2,370 → 1,962~2,023 ms, p2048 4,485~4,970 → 3,969~4,109 ms |
| 950a0ce | 2-bit 호출에서 down epilogue를 background lane 대신 foreground로 (2→4 bit 펼치기 job 뒤에 줄 서지 않게) | DRAIN 13.5 → 4.4 ms, 호출 30.6 → 26.1 ms, p2048 398 → 426 tok/s |
| db645c6 | pre-norm을 HMX scan 안에서, scatter는 호출자 버퍼에 바로 누적 | 2048행에서 DSP heap 46 MB 감소 |
| 8f40a51 외 4개 | padding slot만 있는 16행 단위는 pack 안 함(1024토큰에서 36 MB/call), down scatter 전 출력 zero-fill 생략, staging 크기 정리, read-ahead를 부분적으로라도 queue | 1024토큰 C=16 1810 → 1632 ms, 2048토큰 C=16 3166 → 2954 ms |
| 725b05a | prefill expert reader 스레드 4 → 6 | 1090토큰 1,747 → 1,698 ms (median), 623~629 → 634~650 TPS |

### 4.5 QKV / o-proj

- **q/k head norm + RoPE를 matmul의 dequant epilogue 안에서** (bc36c04): 90 MB qkv 출력을
  한 번 더 왕복하지 않는다. prefill p2048 3,442~3,582 → 3,288~3,325 ms, p4096 7,589 → 7,260 ms.
- **q/k/v 열 slice를 행 순서로 출력에 바로 씀** (8ddabca): host에서 작은 memcpy 264,000번으로
  이어 붙이던 것이 2,970번으로. 132.8 → 97.5 ms.
- **skel이 VTCM 크기만큼 행을 알아서 나눠 돎** (7f8d69c): qkv, o-proj가 각각 호출 하나.
  p2048 qkv 호출 65 → 35, o-proj 95 → 60.
- **attention의 Q를 f32로 바로 받아 DSP에서 u16 양자화, u16 context를 o-proj에 바로 넘김**
  (cc7a637): ARM 쪽 양자화/역양자화 제거. p1024 2,514 / 2,254 → 2,201 / 2,188 ms.
- **pre-norm scratch를 M x K가 아니라 lane당 4행만** (be28fdd): 15~24 MB 잡던 것을 0.6 MB로.
  이것 때문에 p4096에서 AEE_ENOMEMORY가 나던 것이 돌기 시작했다.

### 4.6 Prefill attention

- **row-blocked A8W8 attention 커널** (b3024a8, 47f819e): head 하나의 K^T/V 타일을 VTCM에
  상주시키고, 64행 블록마다 HMX로 QK^T → convert unit으로 int16 readout → 행 전체 integer
  softmax → P'V. score가 VTCM 밖으로 안 나가고 재양자화도 없다. sliding 층(1024 x 1024)에서
  HMX 단계 2.76 → 1.27 ms, softmax 7.5 → 2.8 ms, 호출 18.4 ms.
- **V scale을 32-dim 그룹이 아니라 행당 하나** (c8984f8): P를 그룹마다 8번 양자화하던 것이
  한 번. 호출 22,377 → 19,926 us. 정확도는 0.2~0.4 dB 손해.
- **int8 calibration(K/V abs-max, Q min/max)을 CPU 단일 스레드에서 DSP worker pool로**
  (45b5625), **KV append tile bake를 worker pool로** (00dda0f).
- **임의 길이 context** (0526570): 4096열이 넘으면 fp16으로 떨어지던 것을, 열을 segment로
  나눠 계산하고 epilogue에서 log-sum-exp로 합치게 했다. 22,241 cached + 6,048토큰
  prefill 69.5 → 39.9 s.

### 4.7 TL2 / rotated pack으로 바뀐 뒤

10/15부터 weight pack이 QS2(2-bit palette)에서 native TL2(ternary) + affine으로 바뀌었고,
처음에는 느린 scalar 경로가 드러났다.

| 커밋 | 문제와 수정 | 효과 |
|---|---|---|
| 5ce3054 | TL2 펼치기와 affine 보정이 VTCM을 scalar로 접근하고 있었음 → 벡터화, 펼치기는 background job, 보정은 epilogue worker로 | p2048 prefill 310,313 ms(프로파일 켠 진단 실행) → 3,191 ms median (641.81 TPS) |
| 1e3f661 | strict checkpoint의 Paley-I Hadamard 회전에서 입력 sign을 scalar 루프로 곱하고, 이 모드가 batch dispatch에서 빠져 있었음 → HVX sign pass + batch dispatch | 2K 한국어 요약 prefill 약 353 → 683.70 tok/s (출력 323개 동일) |

### 4.8 Prefill 결과 (S26 Ultra)

| 시점 | p1024 | p2048 |
|---|---|---|
| 10/10 (d4d1eba 전) | 2,549 / 2,716 ms | 398 tok/s (950a0ce 전) |
| 10/11 (4b6d2ad 후) | 1,962~2,023 ms | 3,969~4,109 ms |
| 10/11 (bc36c04 후) | 1,718~1,943 ms | 3,288~3,325 ms |
| 10/14 (8f40a51 후) | 1,632 ms | 2,954 ms |
| 10/16 (725b05a) | 1090토큰 634~650 TPS | 2,924 / 2,932 ms (699~700 TPS) |
| 10/20 strict pack | | 683.70 tok/s (2K 한국어 요약) |

목표였던 700 TPS는 2048토큰에서 경계 수준으로 닿았고, 커밋(b23c6ce)에는 "the 700 TPS
prefill target remains unmet"으로 남아 있다.

---

## 5. Decode 최적화

### 5.1 토큰 하나를 DSP 안에서 끝내기

- **one-PD 상주 그래프**: 처음에는 MoE용 DSP 세션과 FC용 세션 두 개로 돌았고(LFM2 시절
  15~17 tok/s), 세션 사이 hop wakeup이 토큰당 3.0~3.8 ms였다. 세션 하나(one PD)에 FC,
  lm_head, expert pool을 다 넣고 토큰당 dspqueue 패킷 하나로 만들었다.
- **expert miss를 토큰 안에서 처리**: 상주하지 않은 expert로 routing되면, DSP는 miss를
  mailbox에 올리고 가진 expert부터 계산한다. 그동안 ARM이 빠진 expert를 빈 slot에 읽고,
  DSP가 제자리에서 rebind한다. 결과는 전부 상주했을 때와 bit 단위로 같다.
- **id만 돌려받기**: argmax를 DSP에서 한다. 그렇지 않으면 토큰마다 logits 전체(fp32 기준 Gemma4는
  262,144 x 4 B = 1 MiB)를 받아야 한다. sampling이나 banned word가 필요할 때만 logits를 요청한다.
- **prefill과 decode가 FC 가중치 한 벌을 공유** (05e3d1b): decode FC가 prefill용 QS4CX_WH
  handle을 그대로 읽는다. S25에서 Q4M1 FC 사본 255 MiB가 빠졌다.

### 5.2 첫 토큰 setup

첫 decode 토큰은 상주 그래프 초기화와 prefill KV를 decode용 int8 cache로 옮기는(seed) 작업
때문에 3.0~3.2 s가 걸렸고, 그중 2.3~2.5 s가 KV seed였다(p1024).

| 커밋 | 내용 | p1024 first_token_ms | p2048 |
|---|---|---|---|
| 88d9e26 (S25) | seed의 abs-max/append를 DSP worker pool로, fp16→f32 변환은 CPU pool로 | 3.29 → 1.53 s | 5.66 → 2.69 s |
| b75c52e | seed abs-max/append 벡터화 (kv_seed_rpc 428.9 → 92.9 ms) | 1,463 → 1,013 | 2,206 → 1,411 |
| 8baddef | fp16 비트를 그대로 넘기고 DSP에서 넓힘 (host 변환 145 ms → 0) | | |
| 7ed20de | 층마다 RoPE 테이블 복사+비교(30번, 439 ms)하던 것을 이름으로 공유(2번) | → 913 → 506 | → 1,411 → 636 |

### 5.3 Decode attention (ATTN_M1)

| 커밋 | 내용 | 효과 |
|---|---|---|
| edaec7c | decode KV cache를 int8로, scale은 (층, kv head, dim)별 고정 | S25 p1024: ATTN_M1 20.9 → 11.4~11.6 ms/token, cache 440 → 220 MiB |
| 5264090, eb3091e | PV에서 V 행을 64폭 chunk마다 다시 훑던 것을 한 번에 (PV가 호출의 66~70%) | S25: 11.5 → 10.4 ms (p1024), 13.9 → 12.4 ms (p2048) |
| 09ebaf3 | integer decode attention (int16 q, int32 score/PV, 15-bit P) | S26 p1024: 10.36 → 6.73~6.92 ms/token. bit-identical 아님(SNR 27.8~42.1 dB) |
| 46fb91b | sliding 층은 window(1024행)만 KV를 들고 있게 | 6065토큰: sliding decode cache 640 → 106 MiB, 첫 토큰 1082 → 569 ms, 토큰당 143 → 116 ms |

### 5.4 Expert cache: decode 속도의 대부분

decode는 층마다 expert 8개의 가중치를 읽는 시간이 대부분이다. C=16이면 128개 중 16개만
상주하니 miss가 많고, miss 대기가 토큰 시간의 큰 부분을 차지한다.

| 커밋 | 내용 | 효과 |
|---|---|---|
| 1067f41, 961d193 | eviction을 LRU에서 LRFU(사용 빈도를 32토큰 반감기로 감쇠)로, 기본값화 | C=32: p1024 10.83 → 11.93, p2048 10.44 → 12.16 tok/s |
| 8dee795 | LRFU 점수를 한 번에 읽기 | round당 96~109 → 36~39 us |
| b35c5a6 | decode seed 후 CPU K/V 사본 해제 → arena 여유 증가 | S25: MemAvailable +470~520 MiB, decode 11.97 → 12.27, 10.34 → 11.21 tok/s |
| 0d62d96, 690b62d | prefill staging과 KV가 비운 자리만큼, 첫 토큰 후 expert pool을 키움(PD 한도 3,840 MiB - 여유 256 MiB, 목표 층당 34개) | C16 p1024 10.40 → 13.07, C12 p1024 9.75 → 13.28 tok/s (arena 2,240~2,368 MiB) |
| f6e16d3 | decode용 환경 변수 묶음(DMA queue 4개, 다음 층 expert 예측 + read-ahead 등)을 빌트인으로 | C16 p1024 9.6~10.0 → 10.3~10.6 tok/s |
| 9f7f633 | TL2 expert의 gate/up과 down 레코드를 aligned read 한 번으로 합침, 낡은 예측 읽기 취소 | 11.9640 → 12.5731 tok/s, expert 대기 24.74 → 22.62 ms/token |
| 5124689 | TL2 활성 복사 제거, router 가중치 BF16 저장, demand-read 전용 pool, 쓰지 않는 greedy logits 복사 생략 | 12.4661 → 13.2563 tok/s (median, completion hint 켠 상태) |

expert pool을 18개 늘리면 decode가 10.1 → 13.9 tok/s가 되고 arena를 약 1 GB 더 쓴다
(code guide). 결국 decode 속도는 "PD 한도 안에서 expert를 몇 개 상주시킬 수 있느냐"로 많이
정해진다. 그래서 메모리 절감(6장)이 decode 속도와 직결된다.

해봤는데 효과가 없어서 뺀 것: 토큰 중간에 다음 층 expert 예측을 미리 올리는 hint
(533b4df, 10.6 vs 11.4 tok/s로 오히려 느림), 8개 prefetch reader(편차 범위 안).

### 5.5 그 밖의 decode 항목

- **router**: weight 열을 l2fetch로 미리 당기고(ISS 기준 298k → 153k pcycles), weight를
  block-major로 저장해 lane마다 연속 360 KB를 읽게 함.
- **lm_head를 NPU로** (d21823f, 1b541fd): untied QS4CX head를 64개 WH slice로 나눠 decode와
  prefill 마지막 행 모두 DSP에서 계산. 1028토큰에서 decode 13.09 → 13.41 tok/s
  (fc+dense_ffn+lm_head 20.4 → 19.0 ms/token), prefill 약 30 ms 단축. slice 변환 결과를 모델
  옆에 캐시해서 head load가 852 → 124 ms.

### 5.6 Decode 결과

| 조건 | decode |
|---|---|
| 10/13, QS2 pack, 1,028토큰 요약 | 12.3 tok/s (steady 73.7 ms/token) |
| 10/14, QS2 pack, 같은 프롬프트 | 13.8 tok/s |
| 10/14, 1024/2048토큰 C=16, 마지막 64토큰 | 15.0~15.6 tok/s |
| 10/18, TL2 pack, 1024토큰 C=16 | 12.32 tok/s (3회 median) |
| 10/19, rotated pack, 2048 + 128 | 28/29층 affine 11.8 tok/s → 30층 전부 TL2 15.5 tok/s |
| 10/20, strict rotated pack, 2K 한국어 요약 | 19.54 tok/s |
| 30k context (23,063 + 6,641, S25, C=8) | 6.58~6.69 tok/s, ATTN_M1이 토큰 시간의 51% |

마지막 줄에서 보듯 긴 context에서는 병목이 expert에서 attention으로 넘어간다.

---

## 6. 메모리 최적화

| 항목 | before | after | 커밋 |
|---|---|---|---|
| 30k context에서 KV cache 할당 (load 때 층마다 max_seq_len → 첫 실행 때 sliding은 window만) | 908,160행 (6.5 GB) | 69,120행 (495 MB) | a4ed73b, 46fb91b |
| sliding 층 decode cache (6065토큰) | 640 MiB | 106 MiB | 46fb91b |
| decode KV cache int8화 (max_seq 2048) | 440 MiB | 220 MiB | edaec7c |
| page cache 제한 (`NNTR_HTP_PAGE_CACHE=0`, 2048토큰) | peak 8.4 GB (그중 page cache 최대 3.5 GB) | 3.6 GB | 4db63d0 |
| expert 30층 전부 TL2 (28/29층 4-bit 컨테이너 제거) | arena 2,624 MiB, peak 4,723 MB | arena 1,792 MiB, peak 3,669 MB | 4b9d439 |
| TL2 expert slot (QS2 padding 제거) | 1,523,712 B | 1,290,240 B | db84d67 |
| prefill staging 회수 (C16 decode 중 median) | 3,173.25 MiB | 3,124.42 MiB | 63574cc |
| lm_head NPU화 | CPU용 head 사본 396 MiB | 없음 | d21823f |
| FC 가중치 prefill/decode 공유 | Q4M1 사본 255 MiB | 없음 | 05e3d1b |
| CPU K/V 사본 해제 | | MemAvailable +470~520 MiB | b35c5a6 |
| RoPE 테이블 크기 | fp32 1.5 GiB + fp16 768 MiB (OOM) | 실제 길이만큼 | c75e1b8 |
| prefill pre-norm scratch | 15~24 MB/call | 0.6 MB | be28fdd |

10/13 user guide 기준 1,028토큰 요약의 peak(host anon + DSP)는 4.2 GB였다(rebase 전 head는
5.1 GB). page cache 측정에서 보듯 mmap으로 읽으면 OS page cache가 모델 파일을 수 GB 들고
있어서, 이걸 끄면 peak는 크게 줄지만 decode가 9.9 → 7.0 tok/s로 떨어진다. 메모리와 속도
중 무엇을 택할지는 사용처에 따라 정해야 하는 trade-off다.

---

## 7. Prefetch / double buffering 적용 현황

최근 MoE NPU 논문들에서 많이 쓰는 "다음 expert prefetch"와 "double buffering"이 우리에게도
들어가 있는지 정리한다. 결론은 둘 다 들어가 있다. 다만 두 층에서 따로 일어나고, 그중
파일 → DRAM 층은 논문들과 흐름이 다르다.

| 층 | 무엇을 옮기나 | 우리 방식 | 논문과 비교 |
|---|---|---|---|
| DDR → VTCM | 가중치/활성 타일 | DMA로 한 단계 앞서 올리고 계산과 겹침 | 거의 같음 (VTCM 8 MB 때문에 shape별로 다르게 씀) |
| 파일 → DRAM slot | expert 가중치 | prefill은 예측 없이 남은 layer 전부 미리 읽기, decode는 다음 layer router로 예측해서 미리 읽기 | 예측 아이디어는 같은 계열, 비용 구조는 다름 |

### 7.1 DSP 안: DDR → VTCM double buffering

- **decode MoE (VTCM feed)**: VTCM에 gate_up 크기 slab 두 개(G0/G1)를 두고 expert i의
  gate_up을 slab `i & 1`에 넣는다. DMA가 다음 expert를 올리는 동안 HVX GEMV는 현재 expert를
  계산한다. DMA는 queue 4개로 나눠서 보낸다(queue 하나 28~34 GB/s, 넷이면 62 GB/s).
  LFM2 기준 decode +30%.
- **decode FC**: lane마다 자기 VTCM 영역에서 "block b 대기 → block b+1 push → block b 계산"으로
  돈다. 각 lane의 첫 block 전송(slice의 약 1/4)은 숨겨지지 않는다.
- **prefill MoE**: 가중치는 일부러 double buffering하지 않는다(`hexkl_mm_u8i4_moe.c` 첫 주석
  "THE WEIGHTS ARE NOT DOUBLE-BUFFERED"). gate_up과 down을 두 벌씩 두면 10.5 MB인데 VTCM
  arena는 8.3 MB다. 대신 버퍼를 시간차로 재사용한다. expert e의 down이 도는 동안 이미 다 쓴
  gate_up 버퍼에 e+1의 gate_up을 당겨 온다. 활성 쪽은 accumulator staging 버퍼 두 개를 번갈아
  쓰고, block n의 down을 block n+1의 gate_up 다음에 issue해서 HMX가 쉬지 않게 한다
  (software pipelining).
- **prefill attention**: Q를 한 iteration 앞서 DMA로 올리고, 다음 KV head 타일은 VTCM에
  두 벌이 들어갈 때만 미리 올린다.

이 층은 논문들이 말하는 double buffering과 같은 것이라고 말해도 된다.

### 7.2 파일 → DRAM: expert prefetch

**prefill은 예측하지 않는다.** 1024행 이상이면 layer마다 사실상 모든 expert가 쓰이므로,
맞힐 필요 없이 남은 layer의 expert를 reader thread가 차례로 읽어 둔다. horizon은 남은 전체
layer이고, 계산이 끝난 layer의 expert만 evict한다.

| 커밋 | 내용 | 효과 |
|---|---|---|
| d374050 | read-ahead 기본 on (depth 4, reader 4) | (LFM2) cold prefill 2318 → 935 ms |
| 3a7671e | horizon을 남은 전체 layer로 (depth 4로는 C=16에서 preload 뒤 layer를 layer 7이 돼서야 queue) | (LFM2, C=16) 변경 전 cold prefill 799 ms, 하한 약 0.68 s. 변경 후 수치는 없음 |
| 5870fd1 | pool에 일부만 들어가면 layer를 일부라도 queue | C=12에서 1024토큰당 5~141 ms 노출되던 대기 제거 |
| 725b05a | reader 4 → 6 | 1090토큰 1,747 → 1,698 ms (median) |

**decode는 예측해서 읽는다** (8dfc03b, `htp_router_predict.h`, `htp_expert_prefetch.h`).

1. DSP가 layer L을 돌리는 도중에 router 입력 행을 mailbox에 올린다.
2. ARM이 그 행에 layer L+1의 router 가중치를 곱해 top-k를 예측한다. 직전 토큰에서 잰
   layer 간 residual 변화량을 0.5 비율로 더해 보정한다(PREFETCH_RESIDUAL 0.5).
3. rank별 적중률(confidence)로 예측을 한 번 거르고, pool에 없는 expert만 reader 4개가
   여분 DMA slot 32개에 읽는다.
4. 실제 routing에서 miss가 나면 registration RPC 없이 slot만 교환한다. 실제 routing은 항상
   DSP 결과이고, 예측은 I/O 힌트로만 쓴다.

"다음 layer의 gate를 현재 hidden에 미리 적용한다"는 점에서 Pre-gated MoE, EdgeMoE 계열과
같은 아이디어다. 현재는 router가 Gemma의 softmax router일 때만 켜진다(PREDICT=2).

### 7.3 논문들과 다른 점

1. **모바일은 unified memory다.** 논문들은 CPU 메모리나 SSD에서 가속기 메모리로 옮기는
   전송(PCIe 등)을 숨기는 경우가 많다. 우리는 NPU와 CPU가 같은 DRAM을 쓰기 때문에 비용이
   flash read(reader들이 약 3 GB/s 하나를 나눠 씀)와 FastRPC 등록이다. 그래서 효과가 컸던 건
   swap 여러 개를 RPC 하나로 묶기, slot 제자리 rebind, 등록 없는 slot 교환 같은 RPC 줄이기였다.
   LFM2 측정에서 miss 하나가 read 0.39 ms + FastRPC 0.25 ms였고, RPC가 miss 비용의 40%였다.
2. **진짜 제약은 PD 매핑 한도 3,840 MiB다.** prefetch slot도 이 안에 있어야 한다. 그래서 slot을
   prefetch에 쓰는 것보다 상주 expert를 늘리는 쪽이 훨씬 효과가 컸다(pool growth로
   10.40 → 13.07 tok/s).
3. **miss를 토큰 안에서 겹친다.** miss가 나면 DSP는 이미 가진 expert부터 계산하고, 그동안
   ARM이 빠진 expert를 읽고, DSP가 그 slot을 제자리에서 rebind한다. 논문들의 expert 단위
   pipelining과 비슷한 효과다.

### 7.4 예측 prefetch의 실제 효과

| 시도 | 결과 |
|---|---|
| DSP가 예측을 올리고 ARM이 page cache만 데움 (533b4df, S25, C=32) | miss 50개 중 29개를 미리 읽었는데 miss 대기 29.0 → 30.3~31.1 ms, decode 11.4 → 10.6 tok/s. 예측용 router 계산 자체가 비쌌다. 기본 off |
| ARM 예측 + DMA slot 직접 read-ahead (f6e16d3에서 빌트인) | DMA queue 4개, prefill scratch 해제, 8 thread와 묶여서 9.6~10.0 → 10.3~10.6 tok/s (C16 p1024). 예측 하나만 뗀 측정은 없음 |
| prefetch reader 8개 (a7b5ad5) | 편차 범위 안 |
| TL2 gate/up + down을 aligned read 하나로, 낡은 예측 읽기 취소 (9f7f633) | expert 대기 24.74 → 22.62 ms/token, 11.9640 → 12.5731 tok/s |

expert 대기는 여전히 토큰당 약 22.6 ms 남아 있다(TL2 pack, p2048).

### 7.5 설명할 때

"VTCM 수준의 double buffering과 다음 layer expert 예측 prefetch가 적용돼 있다"고 말해도 된다.
다만 "decode 속도 향상의 주원인이 예측 prefetch"라고 하면 과장이다. decode를 실제로 올린 건
PD 한도 안에서 상주 expert를 늘린 것과 weight를 작게 만든 것(TL2)이고, prefetch는 miss 대기를
일부 숨기는 보조 역할이다.

---

## 8. 숫자에 영향을 준 정확도 수정

최적화 중에 잡은 버그 중 앞뒤 수치를 해석할 때 알아야 하는 것들이다.

- **router NaN** (d2782d5): router 누적을 qf32로 하다가 4행 중 3행이 1e37/NaN이 되어 expert
  0~7이 뽑히고 있었다. prompt nll 12.33(균등분포 수준) → 3.536. 이 수정 전의 all-NPU prefill
  시간은 잘못된 routing 위에서 잰 것이다.
- **int8 KV scale이 뒤에 오는 행에 맞춰 커지도록** (eb91939): 전에는 큰 값을 잘라냈다. 대가로
  2k prefill 5.4 → 6.0 s.
- **windowed KV cache의 decode 불일치** (0428a51): max_seq_len > 3072에서 sliding 층이 window
  cache로 바뀌면 9번째 토큰부터 출력이 달라졌다. decode int8 scale을 첫 seed 행들의 |max|로
  정하는데, window cache는 compaction 후 남은 1024행만 seed하기 때문이다. 버려지는 행의
  |max|를 floor로 같이 넘겨 full cache와 bit 단위로 같은 scale이 되게 했다.
- **embedding 행이 찢어지던 문제** (7a6ed2a): page cache를 끄고 멀티스레드로 읽을 때 Q4_0 행
  버퍼를 공유해 출력이 매번 달랐다. 속도 변화 없이 수정.
- **affine FFN decode가 HMX로 가서 fault** (b23c6ce): 수정 후 TL2/affine pack 첫 full
  generation (621.359 prefill / 11.6939 decode TPS).
- **rotated pack에서 decode TL2 경로가 down 회전을 빠뜨림** (91fc867): 이후 rotated 수치의 전제.

---

## 9. 정리와 남은 과제

### 9.1 무엇이 가장 효과가 컸나

Prefill
1. 층을 다 DSP로 옮기고 호출을 합친 것 (fusion) — 1024토큰 4.8 s → 3.8 s (S25)
2. 블록 텐서를 DSP에 두고 staging 복사 제거 — p2048 memcpy 402 → 39 ms
3. MoE 행 양자화 한 번 — p2048 약 4.6 s → 4.0 s
4. norm/RoPE를 matmul epilogue로 — p2048 약 3.5 s → 3.3 s
5. padding skip + staging 정리 — 3166 → 2954 ms
6. (pack 변경 후) TL2 펼치기 벡터화, Paley sign HVX화 — 353 → 683.70 tok/s 회복

Decode
1. 토큰 하나를 상주 그래프로 DSP 안에서 끝내는 구조 (기반)
2. expert pool을 PD 여유만큼 키우기 — 10.40 → 13.07 tok/s
3. int8 KV + PV 단일 walk + integer attention — ATTN_M1 20.9 → 10.4 ms (S25), S26에서 10.36 → 6.73 ms
4. 첫 토큰 setup — 1,463 → 506 ms
5. weight를 TL2로 (읽는 바이트 감소) — 11.8 → 15.5 tok/s

### 9.2 참고: 같은 SoC의 GPU(Adreno 840) 경로

같은 PR에 들어 있는 OpenCL 시리즈(다른 작업자)의 Gemma4 26B-A4B 수치다. expert가 TQ2CX
ternary라 weight pack이 다르므로 직접 비교는 아니고 참고용이다.

| | Adreno 840 GPU | Hexagon NPU (이 문서) |
|---|---|---|
| prefill 1K | 약 174 → 388 TPS | 627~630 tok/s |
| decode | 14~17 → 25.3 TPS (resident), 20.3 TPS (32 slot streaming) | 13~15 tok/s (QS2/TL2), 19.54 tok/s (strict pack) |
| peak RSS | 7.8 GB (resident), 3.4 GB (streaming) | 3.6~4.2 GB |

prefill은 NPU가 확실히 빠르고, decode는 현재 GPU가 더 빠르거나 비슷하다. 발표할 때
"NPU가 전부 빠르다"고 하면 안 되고, prefill과 메모리에서 이점이 있다고 말하는 게 맞다.

### 9.3 남은 것

- **decode 16 tok/s 목표**: 10/15 decode 개선 시리즈에서 목표 미달로 기록. strict pack에서는
  19.54지만 pack이 다르다.
- **static activation 모드**: 10/20 static grid 경로는 prefill 82.16 / decode 16.86 tok/s(1회)로
  dynamic(677.82 / 19.42)보다 훨씬 느리다. 또 KV나 다른 QDQ 경계는 아직 덮지 않는다.
- **에이전트 품질 게이트 실패**: PetCare sub-agent TC가 기대한 tool call
  (`turnOnPetMonitoring`) 대신 `launchPetMonitoring`을 부른다. dynamic 모드도 같다.
- **긴 context decode**: 30k에서 ATTN_M1이 토큰 시간의 51%, 실효 대역폭 약 4.2 GB/s로
  DDR 대역폭에 한참 못 미친다. 커널 여지가 있다(추정, 미측정).
- **메모리가 max_seq_len에 비례해 load 때 잡히는 부분**이 아직 남아 있다(30272에서 peak 5.25 GB).
- **원인 미확정 이슈**: snapshot/demo에서 expert-load rebind 실패 1회(11회 반복 재현 안 됨).

---

## 부록 A. 측정 조건 주의

- 장치가 섞여 있다. 10/08 이전과 일부 커밋은 S25 Ultra(v79), 나머지는 S26 Ultra(v81)다.
  표에 S25라고 적은 것만 S25이고, 나머지는 S26이다.
- weight pack이 시기마다 다르다: 10/14까지 QS2CX_WH + QS4CX + Q4_0 (7.5 GB), 10/15부터
  native TL2 + affine (6.6 GB), 10/16부터 rotated, 10/20 strict rotated + Paley.
  서로 다른 pack의 숫자를 한 줄로 이어 붙여 "개선"이라고 하면 안 된다.
- 폰 온도 제어가 안 된 단발 측정이 많고 편차가 10~20%다. 커밋들은 주로 번갈아 3회
  실행한 값을 쓴다.
- C는 층당 상주 expert 수, pNNNN은 프롬프트 토큰 수, "last 64"는 마지막 64토큰 decode 속도.
- 출처 문서: `docs/gemma4-htp-user-guide.md`, `docs/gemma4-htp-code-guide.md`,
  `docs/gemma4-htp-tl2-guide.md`, `docs/gemma4-htp-kv-cache-s25u.md`,
  `docs/gemma4-htp-lmhead-npu.md`.

## 부록 B. 핵심 커밋 목록

| 분류 | 커밋 |
|---|---|
| prefill fusion | 2cb8f82, f628465, b78b889, a14e326, f8d8cb7, 3d77223, 02d899e, a2846b7, cacc9a8 |
| prefill staging | 6bd991e, 3966f84, 5f53003, 5870fd1 |
| prefill MoE | 4b6d2ad, 950a0ce, db645c6, 8f40a51, 821dc8d, 725b05a |
| prefill QKV/o-proj | bc36c04, 8ddabca, 7703461, 7f8d69c, a9842cd, cc7a637, be28fdd |
| prefill attention | b3024a8, 47f819e, c8984f8, 030c007, 45b5625, 00dda0f, 0526570, 98869e9, 523c840 |
| router | 6606ec0, 141d1b1, 3d7661f, e6522ee, d2782d5, 365de62, dd1a2a8 |
| TL2/rotated | 5ce3054, 256ac9f, db84d67, e9aa86a, 1e3f661, 45193d1, 4b9d439 |
| decode 구조 | 687932c, 5d6b749, 99b71f4, 05e3d1b, c3287d9, f6e16d3 |
| 첫 토큰 | fa65875, 88d9e26, b75c52e, 8baddef, 7ed20de, 4e307ef |
| decode attention | edaec7c, 5264090, eb3091e, 09ebaf3, 46fb91b, 0428a51 |
| expert cache | 1067f41, 961d193, 8dee795, 0d62d96, 690b62d, 9f7f633, 5124689, b35c5a6 |
| prefetch / double buffering | 434e262, 8f1ad33, 74a24db, 8dfc03b, 533b4df, d374050, 3a7671e, 5870fd1, 725b05a, f6e16d3 |
| lm_head | d21823f, 1b541fd |
| 메모리 | a4ed73b, 4db63d0, 63574cc, 74e590c, c75e1b8 |
