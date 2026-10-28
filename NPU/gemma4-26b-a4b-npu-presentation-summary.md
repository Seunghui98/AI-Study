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

## 2. Prefill

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

## 3. Decode

구조 : DSP 상주 그래프
- decode step 전체(542 op)를 NPU에 그래프로 상주
- CPU → position 전달 / NPU → 다음 token id만 반환
- 토큰당 메시지 1번, 그 사이 CPU idle

첫 토큰 준비 (prefill KV → decode int8 cache 변환 + 그래프 초기화)
- 변환 작업 병렬화 / 벡터화
- RoPE 테이블 30번 전송 → 1번 공유
- 1,463 → 506 ms

Attention
- KV cache int8 + 정수 연산 attention
- 토큰당 attention 20.9 → 6.7 ms (S25U → S26U)

expert cache (decode 시간 대부분 = expert weight 읽기)
- 첫 토큰 후 남는 메모리로 상주 expert 16 → 최대 34개 : 10.4 → 13.1 TPS
- expert 교체 정책 LRU → LRFU
- 다음 layer expert 예측 prefetch (보조, 단독 효과 작음)
- weight 2-bit → ternary(1.67 bit) : 읽는 양 감소, 11.8 → 15.5 TPS

---

## 4. 메모리

```
VTCM 8 MiB (prefill MoE)          VTCM 8 MiB (decode MoE)
├─ 입력 64행        176 KiB       ├─ expert 버퍼 A   968 KiB  ← 짝수 expert
├─ gate_up weight   1.89 MiB      ├─ expert 버퍼 B   968 KiB  ← 홀수 expert
├─ down weight      0.95 MiB      │   한쪽 계산 중 다른 쪽에 다음 expert DMA
├─ 중간/결과 버퍼    1.8 MiB       └─ 나머지 여유
└─ 합계 4.8 MiB
```

expert 1개 : 1.45 MiB (2-bit) / 1.23 MiB (ternary) / 참고 LFM2 5.25 MiB

| 항목 | 전 | 후 |
|---|---|---|
| 3만 토큰 KV cache | 6.5 GB | 495 MB |
| sliding 층 decode cache | 640 MiB | 106 MiB |
| expert 전부 ternary | peak 4.7 GB | 3.7 GB |
| lm_head NPU (CPU 사본 제거) | 396 MiB | 0 |

---

## 5. GPU 비교 (같은 칩, weight 포맷 다름)

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
